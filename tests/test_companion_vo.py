"""Конвейер VO компаньон-компьютера: синхронизация, высота, скорость FRD."""
import math

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from src.camera_rotation_compensation import CameraMountCalibration
from src.companion_vo import CompanionVo, History, TimeOffset, VerticalVelocityGate
from src.continuous_visual_odometry import ContinuousPlanarOdometry
from src.ground_visual_motion import CameraCalibration
from src.surface_depth import SurfaceDepth
from src.velocity_uncertainty import VelocityErrorAssumptions

IMAGE_TO_BODY = ((0.0, -1.0), (1.0, 0.0))
BODY_FROM_CAMERA = ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
F_PX = 240.0
OFFSET_US = 7_000_000  # часы автопилота впереди часов компаньона


def _texture():
    rng = np.random.default_rng(5)
    img = np.zeros((900, 900), np.float32)
    for cells, weight in ((12, 0.6), (45, 1.0), (150, 0.8)):
        img += weight * cv2.resize(rng.random((cells, cells)).astype(np.float32), (900, 900),
                                   interpolation=cv2.INTER_CUBIC)
    img = (img - img.min()) / (img.max() - img.min())
    return (255 * img).astype(np.uint8)


def _pipeline(window_pairs=1, surface=None, vertical_gate=None):
    cal = CameraCalibration(F_PX, F_PX, 160.0, 120.0, IMAGE_TO_BODY)
    vo = ContinuousPlanarOdometry(
        cal, camera_mount=CameraMountCalibration(BODY_FROM_CAMERA),
        velocity_error_assumptions=VelocityErrorAssumptions(0.5, 0.05, 0.002, 0.02, 0.01,
                                                            velocity_model_floor_m_s=0.05),
        max_tilt_rad=0.25, surface=surface)
    pipe = CompanionVo(vo, window_pairs=window_pairs, vertical_gate=vertical_gate)
    for k in range(5):  # обмен TIMESYNC, задержка круга 2-6 мс
        sent = 1_000_000 + k * 200_000
        rtt = 2000 + 1000 * k
        pipe.clock.add(sent, sent + rtt // 2 + OFFSET_US, sent + rtt)
    return pipe


def test_time_offset_uses_minimum_round_trip_and_rejects_slow_samples():
    clock = TimeOffset()
    assert not clock.valid
    clock.add(0, 5_000 + 900, 1_000)          # rtt 1 мс -> смещение 5 мс + 0.4
    clock.add(10_000, 10_000 + 5_000 + 20_000, 40_000)  # rtt 30 мс: асимметрия
    clock.add(20_000, 20_000 + 9_999_999, 120_000)      # rtt 100 мс: отброшен
    clock.add(50_000, 50_000 + 5_000 + 1_000, 52_000)
    assert clock.valid
    assert clock.offset_us == 5_400
    with pytest.raises(ValueError):
        TimeOffset().offset_us


def test_history_interpolates_only_inside_a_bounded_gap():
    h = History(max_gap_us=100)
    h.add(1000, 1.0)
    h.add(1100, 3.0)
    h.add(1050, 9.0)  # немонотонный отсчет отбрасывается
    assert h.bracket(1050) == (1.0, 3.0, 0.5)
    assert h.bracket(999) is None and h.bracket(1101) is None
    h.add(1300, 5.0)
    assert h.bracket(1200) is None  # разрыв 200 мкс > 100


def test_forward_and_climb_give_body_frd_velocity_from_measurements():
    """Вперед 1 м/с, подъем 0.5 м/s, курс 90 градусов, без наклона."""
    texture = _texture()
    pipe = _pipeline()
    yaw = math.pi / 2
    q = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
    dt_us = 66_667
    outs = []
    for k in range(8):
        t_cam = 2_000_000 + k * dt_us
        height = 3.0 + 0.5 * k * dt_us * 1e-6
        # Отсчеты автопилота каждые 10 мс вокруг кадра, в его часах.
        for j in range(-8, 9):
            t_ap = t_cam + OFFSET_US + j * 10_000
            pipe.attitude.add(t_ap, q)
            pipe.distance.add(t_ap, 3.0 + 0.5 * (t_ap - OFFSET_US - 2_000_000) * 1e-6)
        assert pipe.ready_for(t_cam)
        # Сдвиг вперед на 1 м/с: изображение земли уходит вниз (+v).
        shift_px = 1.0 * k * dt_us * 1e-6 * F_PX / height
        row = 400 - shift_px
        crop = cv2.getRectSubPix(texture, (320, 240), (450.0, row + 120.0))
        # Масштаб от высоты не моделируется: подъем мал на 8 кадрах.
        out = pipe.process(t_cam, crop)
        if out is not None:
            outs.append(out)
    assert len(outs) >= 5, pipe.counts
    vx = np.median([o.velocity_frd[0] for o in outs])
    vy = np.median([o.velocity_frd[1] for o in outs])
    vz = np.median([o.velocity_frd[2] for o in outs])
    assert vx == pytest.approx(1.0, abs=0.12)
    assert abs(vy) < 0.1
    assert vz == pytest.approx(-0.5, abs=0.05)   # подъем: FRD z вниз
    # Курс 90°: вперед по корпусу = на восток.
    assert np.median([o.vo.ve_m_s for o in outs]) == pytest.approx(1.0, abs=0.12)
    o = outs[-1]
    assert o.variance_frd[2] == pytest.approx(2 * 0.02 ** 2 / (dt_us * 1e-6) ** 2, rel=1e-6)
    assert o.time_us == (o.vo.interval_start_us + o.vo.timestamp_us) // 2


@pytest.mark.parametrize("surface", [False, True])
def test_window_sends_the_mean_over_contiguous_pairs(surface):
    """Speed alternating 0.4 / 1.6 m/s per pair: the 2-pair window sends 1 m/s.

    With a flat DSM (and a position) the feature depths equal the height.
    """
    texture = _texture()
    dsm = SurfaceDepth(np.full((200, 200), 7.0, np.float32), (50.0, -50.0), 0.5) if surface else None
    pipe = _pipeline(window_pairs=2, surface=dsm)
    q = (1.0, 0.0, 0.0, 0.0)
    dt_us = 66_667
    outs, travelled = [], 0.0
    for k in range(9):
        t_cam = 2_000_000 + k * dt_us
        for j in range(-8, 9):
            t_ap = t_cam + OFFSET_US + j * 10_000
            pipe.attitude.add(t_ap, q)
            pipe.distance.add(t_ap, 3.0)
        if k:
            travelled += (0.4 if k % 2 else 1.6) * dt_us * 1e-6
        crop = cv2.getRectSubPix(texture, (320, 240), (450.0, 520.0 - travelled * F_PX / 3.0))
        out = pipe.process(t_cam, crop, position_ne_m=(0.0, 0.0) if surface else None)
        if out is not None:
            outs.append(out)
    full = [o for o in outs if o.pairs_in_window == 2]
    assert len(full) >= 5, pipe.counts
    assert outs[0].pairs_in_window == 1
    for o in full:
        assert o.velocity_ne_sent[0] == pytest.approx(1.0, abs=0.12)
        assert o.velocity_frd[0] == pytest.approx(1.0, abs=0.12)
        assert o.time_us == o.vo.timestamp_us - dt_us
    # The last pair itself still alternates (anchor propagation uses it).
    assert max(o.vo.vn_m_s for o in full) - min(o.vo.vn_m_s for o in full) > 0.8


def test_no_measurement_without_time_sync_or_rangefinder():
    texture = _texture()
    cal = CameraCalibration(F_PX, F_PX, 160.0, 120.0, IMAGE_TO_BODY)
    pipe = CompanionVo(ContinuousPlanarOdometry(cal, camera_mount=CameraMountCalibration(BODY_FROM_CAMERA)))
    crop = texture[:240, :320].copy()
    assert not pipe.ready_for(1_000_000)
    assert pipe.process(1_000_000, crop) is None
    assert pipe.counts == {"NO_TIME_SYNC": 1}
    pipe = _pipeline()
    for j in range(20):
        pipe.attitude.add(1_900_000 + OFFSET_US + j * 10_000, (1.0, 0.0, 0.0, 0.0))
    assert not pipe.ready_for(2_000_000)  # нет дальномера
    assert pipe.process(2_000_000, crop) is None
    assert pipe.counts["NO_ATTITUDE_OR_RANGE"] == 1


def test_no_measurement_below_minimum_height():
    texture = _texture()
    pipe = _pipeline()
    for j in range(-20, 21):
        t_ap = 2_000_000 + OFFSET_US + j * 10_000
        pipe.attitude.add(t_ap, (1.0, 0.0, 0.0, 0.0))
        pipe.distance.add(t_ap, 0.3)
    assert pipe.process(2_000_000, texture[:240, :320].copy()) is None
    assert pipe.counts == {"BELOW_MIN_HEIGHT": 1}


def test_velocity_jump_gate_rejects_impossible_jumps_and_recovers():
    from src.companion_vo import VelocityJumpGate
    g = VelocityJumpGate(max_accel_m_s2=3.7, noise_m_s=0.6, max_hold_s=1.0)
    t = 0
    assert g.check(t, (0.0, 0.0))
    t += 130_000
    assert g.check(t, (0.5, 0.0))                 # 0.5 m/s in 0.13 s: within 3.7*0.13 + 0.6
    t += 130_000
    assert not g.check(t, (1.8, 1.0))             # the series P outlier: 1.6 m/s jump
    v, var = g.held_velocity(t)
    assert v == (0.5, 0.0) and var == pytest.approx((3.7 * 0.13) ** 2 + 0.36)
    t += 130_000
    assert g.check(t, (0.6, 0.1))                 # back to plausible: compared with the last accepted
    assert not g.check(t + 130_000, (float("nan"), 0.0))
    # After max_hold_s without an accepted sample the next one is taken as is.
    assert not g.check(t + 200_000, (9.0, 0.0))
    assert g.check(t + 1_200_000, (3.0, 0.0))


def test_dsm_altitude_is_sampled_at_the_rangefinder_hit_point():
    # Rolled 0.2 rad (right side down): the body-down rangefinder hits
    # ~6 m west, on a 20 m roof; the ground under the vehicle is at 0 m.
    grid = np.zeros((200, 200), np.float32)
    grid[:, :92] = 20.0                       # west of -4 m
    dsm = SurfaceDepth(grid, (50.0, -50.0), 0.5)
    pipe = _pipeline(surface=dsm)
    roll = 0.2
    q = (math.cos(roll / 2), math.sin(roll / 2), 0.0, 0.0)
    slant = 30.0 / math.cos(roll)             # 30 m to the roof along the body axis
    t_cam = 2_000_000
    for j in range(-8, 9):
        t_ap = t_cam + OFFSET_US + j * 10_000
        pipe.attitude.add(t_ap, q)
        pipe.distance.add(t_ap, slant)
    seen = []
    pipe.vo.process = lambda frame: seen.append(frame) or pipe.vo.reject_frame(frame.timestamp_us, "TEST")
    pipe.process(t_cam, _texture()[:240, :320], position_ne_m=(0.0, 0.0))
    assert seen[0].altitude_m == pytest.approx(50.0, abs=0.05)


@pytest.mark.parametrize("gated", [False, True])
def test_range_step_does_not_reach_the_vertical_velocity_when_gated(gated):
    """A 0.3-m rangefinder step (roof edge) between two frames: 4.5 m/s."""
    texture = _texture()
    pipe = _pipeline(vertical_gate=VerticalVelocityGate() if gated else None)
    q = (1.0, 0.0, 0.0, 0.0)
    dt_us = 66_667
    outs = []
    for k in range(8):
        t_cam = 2_000_000 + k * dt_us
        for j in range(-8, 9):
            t_ap = t_cam + OFFSET_US + j * 10_000
            pipe.attitude.add(t_ap, q)
            pipe.distance.add(t_ap, 3.0 + (0.3 if t_ap - OFFSET_US >= 2_000_000 + 5 * dt_us - 5_000 else 0.0))
        crop = cv2.getRectSubPix(texture, (320, 240), (450.0, 520.0 - 0.5 * k * dt_us * 1e-6 * F_PX / 3.0))
        out = pipe.process(t_cam, crop)
        if out is not None:
            outs.append(out)
    vz = [o.velocity_frd[2] for o in outs]
    assert len(vz) >= 5, pipe.counts
    # The step is smeared over the 10-ms range samples: -3 m/s, then -1.5.
    if gated:
        # What passes is bounded by the vertical acceleration over the hold
        # plus the noise (9.81 x 0.133 s + 0.7 m/s), not tens of m/s.
        assert max(abs(v) for v in vz) <= 9.81 * 2 * dt_us * 1e-6 + 0.7
        assert pipe.counts["VO_VERTICAL_REJECTED"] >= 1
        assert min(vz) > -2.9
        assert min(o.variance_frd[2] for o in outs) == pytest.approx(2.5 ** 2)
    else:
        assert min(vz) < -2.9


def test_vertical_gate_limits_and_hold():
    g = VerticalVelocityGate()
    assert g.held(0) == (0.0, 64.0)                 # nothing accepted yet
    assert g.check(1_000_000, 0.5)
    assert not g.check(1_066_000, 30.0)             # range step
    assert not g.check(1_132_000, 9.0)              # beyond the absolute bound
    vd, var = g.held(1_500_000)
    assert vd == 0.5 and var == pytest.approx((9.81 * 0.5) ** 2 + 0.7 ** 2)
    assert g.check(1_200_000, 1.5)                  # 0.5 + 9.81 x 0.2 + 0.7 allows it
    assert not g.check(3_000_000, 8.5)              # after the hold: still bounded
    assert g.check(3_000_000, 6.0)                  # after the hold: taken as is
    with pytest.raises(ValueError):
        VerticalVelocityGate(max_accel_m_s2=0.0)
