"""MSCKF на синтетике: наземные точки, полет с ускорениями, неизвестные смещения ИИМ."""
import math

import numpy as np
import pytest

from src.error_state_inertial_filter import ImuNoiseDensity
from src.imu_preintegration import ImuReading, rotation_body_to_ned
from src.msckf_vio import Msckf, PinholeCamera, _chi2_95

T0 = 1.0
NOISE = ImuNoiseDensity(gyro_rad_s_sqrt_hz=0.002, accel_m_s2_sqrt_hz=0.02,
                        gyro_bias_walk_rad_s2_sqrt_hz=1e-5, accel_bias_walk_m_s3_sqrt_hz=1e-4)
CAM = PinholeCamera(400.0, 400.0, 320.0, 240.0, ((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
                    (0.1, 0.0, 0.05), pixel_sigma=0.5)


def _quat(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = (math.cos(roll / 2), math.sin(roll / 2), math.cos(pitch / 2),
                              math.sin(pitch / 2), math.cos(yaw / 2), math.sin(yaw / 2))
    return np.array([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy])


def _truth(t):
    w = 0.35
    p = np.array([12 * math.sin(w * t), 8 * math.sin(2 * w * t), -30 + 2 * math.sin(w * t)])
    v = np.array([12 * w * math.cos(w * t), 16 * w * math.cos(2 * w * t), 2 * w * math.cos(w * t)])
    a = np.array([-12 * w * w * math.sin(w * t), -32 * w * w * math.sin(2 * w * t), -2 * w * w * math.sin(w * t)])
    q = _quat(0.05 * math.sin(0.7 * t), 0.04 * math.cos(0.5 * t), 0.3 + 0.05 * t)
    return p, v, a, q


def _omega(t, dt=1e-4):
    q0, q1 = _truth(t - dt)[3], _truth(t + dt)[3]
    # ω_body = 2 vec(q0^-1 q1) / (2 dt)
    w0, v0 = q0[0], -q0[1:]
    w1, v1 = q1[0], q1[1:]
    vec = w0 * v1 + w1 * v0 + np.cross(v0, v1)
    return vec / dt


def _simulate(seconds=24.0, imu_hz=200, cam_hz=10, gyro_bias=(0.002, -0.0015, 0.001),
              accel_bias=(0.04, -0.03, 0.05), seed=4):
    rng = np.random.default_rng(seed)
    points = np.column_stack([rng.uniform(-60, 60, 6000), rng.uniform(-60, 60, 6000),
                              rng.normal(0.0, 0.3, 6000)])
    r_bc = np.asarray(CAM.body_from_camera)
    p_bc = np.asarray(CAM.camera_position_body_m)
    imu, frames, truth = [], [], []
    n = int(seconds * imu_hz)
    for k in range(0, n + 1):
        t = T0 + k / imu_hz
        p, v, a, q = _truth(t)
        r = rotation_body_to_ned(q)
        f = r.T @ (a - np.array([0, 0, 9.80665])) + np.array(accel_bias)
        w = _omega(t) + np.array(gyro_bias)
        t_us = int(round(t * 1e6))
        imu.append(ImuReading(t_us, tuple(w + rng.normal(0, 1e-4, 3)), tuple(f + rng.normal(0, 2e-3, 3))))
        if k and k % (imu_hz // cam_hz) == 0:
            r_wc = r @ r_bc
            p_wc = p + r @ p_bc
            pc = (points - p_wc) @ r_wc  # rows: R_wc^T (x - p_wc)
            uv = np.column_stack([CAM.fx * pc[:, 0] / pc[:, 2] + CAM.cx, CAM.fy * pc[:, 1] / pc[:, 2] + CAM.cy])
            vis = np.where((pc[:, 2] > 1) & (uv[:, 0] > 0) & (uv[:, 0] < 640) & (uv[:, 1] > 0) & (uv[:, 1] < 480))[0]
            vis = vis[:120]
            feats = {int(i): tuple(uv[i] + rng.normal(0, 0.5, 2)) for i in vis}
            frames.append((t_us, feats))
            truth.append((t_us, p, v))
    return imu, frames, truth


def _run(use_camera=True, **kw):
    imu, frames, truth = _simulate(**kw)
    p0, v0, _, q0 = _truth(T0)
    f = Msckf(camera=CAM, noise=NOISE, quaternion_body_to_ned=tuple(q0),
              initial_position_ned_m=tuple(p0), initial_velocity_ned_m_s=tuple(v0))
    fi = 0
    errors = []
    for reading in imu:
        f.predict(reading)
        if fi < len(frames) and frames[fi][0] == reading.timestamp_us:
            if use_camera:
                f.process_frame(*frames[fi])
            errors.append(np.linalg.norm(f.p - truth[fi][1]))
            fi += 1
    return f, errors


@pytest.fixture(scope="module")
def runs():
    return _run(True), _run(False)


def test_msckf_bounds_drift_that_inertial_only_cannot(runs):
    (vio, err_vio), (_, err_imu) = runs
    assert vio.stats.updates > 150 and vio.stats.features_used > 2000
    assert err_imu[-1] > 15.0                  # biased IMU alone: tens of metres
    assert err_vio[-1] < 0.05 * err_imu[-1]
    assert max(err_vio) < 1.0


def test_msckf_estimates_imu_biases(runs):
    (vio, _), _ = runs
    assert np.allclose(vio.bg, (0.002, -0.0015, 0.001), atol=5e-4)
    assert abs(vio.ba[2] - 0.05) < 0.02


def test_msckf_covariance_is_consistent(runs):
    (vio, _), _ = runs
    _, _, truth = _simulate()
    err = vio.p - truth[-1][1]
    nees = float(err @ np.linalg.solve(vio.cov[0:3, 0:3], err))
    assert nees < _chi2_95(3) * 3  # loose: single sample


def test_frames_must_align_with_imu():
    f = Msckf(camera=CAM, noise=NOISE, quaternion_body_to_ned=(1.0, 0.0, 0.0, 0.0))
    f.predict(ImuReading(1000, (0, 0, 0), (0, 0, -9.80665)))
    with pytest.raises(ValueError):
        f.process_frame(2000, {})


def test_klt_tracker_keeps_ids_on_a_shifted_texture():
    cv2 = pytest.importorskip("cv2")
    from src.feature_tracker import KltTracker
    rng = np.random.default_rng(3)
    tex = cv2.GaussianBlur((rng.random((600, 800)) * 255).astype(np.uint8), (0, 0), 2.0)
    tracker = KltTracker(max_features=80)
    first = tracker.track(tex[50:530, 50:690].copy())
    second = tracker.track(tex[53:533, 48:688].copy())  # camera moved: texture shifts (+2, -3) px
    common = set(first) & set(second)
    assert len(common) > 50
    shifts = np.array([np.subtract(second[i], first[i]) for i in common])
    assert np.allclose(np.median(shifts, axis=0), (2.0, -3.0), atol=0.1)


def test_linear_vi_initialization_recovers_velocity_without_gnss():
    from src.vi_initialization import initialize
    imu, frames, truth = _simulate(seconds=4.0, accel_bias=(0.04, -0.03, 0.05), gyro_bias=(0.0, 0.0, 0.0))
    window = frames[:20]  # 2 s at 10 Hz
    q0 = tuple(_truth(window[0][0] * 1e-6)[3])
    # Rangefinder: height above the (flat, +-0.3 m) ground, 2 % noise.
    rng = np.random.default_rng(9)
    heights = {t: float(-truth[k][1][2]) * (1 + rng.normal(0, 0.02)) for k, (t, _) in enumerate(window)}
    res = initialize(imu, window, CAM, q0, heights_agl_m=heights)
    assert res.ok, res
    v_true_end = truth[19][2]
    assert np.allclose(res.velocity_end_ned_m_s, v_true_end, atol=0.1)
    assert abs(res.accel_bias_body_m_s2[2] - 0.05) < 0.03
    assert 25 < res.median_depth_m < 40
    assert max(abs(x) for x in res.gravity_tilt_deg) < 0.5


def test_vi_initialization_without_height_refuses_or_is_accurate():
    from src.vi_initialization import initialize
    imu, frames, truth = _simulate(seconds=4.0, accel_bias=(0.04, -0.03, 0.05), gyro_bias=(0.0, 0.0, 0.0))
    window = frames[:20]
    res = initialize(imu, window, CAM, tuple(_truth(window[0][0] * 1e-6)[3]))
    # Horizontal bias and gravity tilt look alike over 2 s: weak scale.
    if res.ok:
        assert np.allclose(res.velocity_end_ned_m_s, truth[19][2], atol=3 * res.sigma_velocity_m_s)
    else:
        assert res.status == "INSUFFICIENT_EXCITATION"


def test_linear_vi_initialization_refuses_without_acceleration():
    from src.vi_initialization import initialize
    from src.imu_preintegration import ImuReading
    rng = np.random.default_rng(1)
    pts = np.column_stack([rng.uniform(-30, 30, 800), rng.uniform(-30, 30, 800), np.zeros(800)])
    r_bc = np.asarray(CAM.body_from_camera)
    v = np.array([2.0, 0.0, 0.0])
    imu = [ImuReading(1_000_000 + k * 5000, (0.0, 0.0, 0.0), (0.0, 0.0, -9.80665)) for k in range(0, 501)]
    frames = []
    for k in range(1, 21):
        t = k * 0.1
        p = np.array([0.0, 0.0, -30.0]) + v * t
        pc = (pts - p - r_bc @ np.zeros(3)) @ r_bc
        uv = np.column_stack([CAM.fx * pc[:, 0] / pc[:, 2] + CAM.cx, CAM.fy * pc[:, 1] / pc[:, 2] + CAM.cy])
        vis = np.where((uv[:, 0] > 0) & (uv[:, 0] < 640) & (uv[:, 1] > 0) & (uv[:, 1] < 480))[0][:100]
        frames.append((1_000_000 + k * 100_000, {int(i): tuple(uv[i]) for i in vis}))
    cam = PinholeCamera(CAM.fx, CAM.fy, CAM.cx, CAM.cy, CAM.body_from_camera, (0.0, 0.0, 0.0), 0.5)
    res = initialize(imu, frames, cam, (1.0, 0.0, 0.0, 0.0))
    assert not res.ok and res.status in ("INSUFFICIENT_EXCITATION", "DEGENERATE_GEOMETRY")
