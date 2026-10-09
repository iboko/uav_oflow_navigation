"""Регрессионные тесты потоковой оценки относительного перемещения."""
from math import pi
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from src.continuous_visual_odometry import ContinuousPlanarOdometry, VisualFrame
from src.ground_visual_motion import CameraCalibration


def _calibration():
    # Конвенция осей только для синтетического стенда.
    return CameraCalibration(1400.0, 1400.0, 320.0, 240.0,
                             ((1.0, 0.0), (0.0, 1.0)))


def _terrain():
    rng = np.random.default_rng(91)
    a = np.zeros((480, 640), np.uint8)
    for _ in range(460):
        x, y = rng.integers([18, 18], [622, 462])
        cv2.circle(a, (int(x), int(y)), 2, int(rng.integers(120, 250)), -1)
    return a


def _translate(img, dx, dy=0):
    return cv2.warpAffine(
        img, np.float32([[1.0, 0.0, dx], [0.0, 1.0, dy]]),
        (img.shape[1], img.shape[0]),
    )


def _frame(img, t_us, h=100.0, yaw=0.0, roll=0.0, pitch=0.0):
    return VisualFrame(t_us, img, h, yaw, roll, pitch)


def test_three_frames_integrate_signed_relative_motion_100m():
    terrain = _terrain()
    odom = ContinuousPlanarOdometry(_calibration())
    first = odom.process(_frame(terrain, 1_000_000))
    second = odom.process(_frame(_translate(terrain, 2.0), 1_200_000))
    third = odom.process(_frame(_translate(terrain, 4.0), 1_400_000))
    assert first.status == "INITIALIZED" and not first.accepted
    assert second.accepted and third.accepted
    assert second.segment_id == third.segment_id == first.segment_id
    assert third.north_m == pytest.approx(-4.0 * 100 / 1400, abs=0.09)
    assert abs(third.east_m) < 0.09
    assert third.resolution_proxy_m > second.resolution_proxy_m > 0.0
    assert third.inlier_count >= 20
    assert third.inlier_ratio > 0.6


def test_time_gap_starts_unconnected_segment_without_integrating_displacement():
    terrain = _terrain()
    odom = ContinuousPlanarOdometry(_calibration())
    odom.process(_frame(terrain, 1_000_000))
    good = odom.process(_frame(_translate(terrain, 2), 1_200_000))
    assert good.accepted
    gap = odom.process(_frame(_translate(terrain, 8), 1_900_000))
    assert not gap.accepted and gap.status == "TIME_GAP_REINITIALIZED"
    assert gap.segment_id == good.segment_id + 1
    assert gap.north_m == 0.0 and gap.east_m == 0.0
    recovered = odom.process(_frame(_translate(terrain, 10), 2_100_000))
    assert recovered.accepted
    assert recovered.segment_id == gap.segment_id
    assert recovered.north_m == pytest.approx(-2 * 100 / 1400, abs=0.06)


def test_loss_of_texture_is_not_silently_interpolated():
    terrain = _terrain()
    odom = ContinuousPlanarOdometry(_calibration())
    odom.process(_frame(terrain, 1_000_000))
    result = odom.process(_frame(np.full_like(terrain, 130), 1_200_000))
    assert not result.accepted
    assert result.status.startswith("TRACK_LOST:")
    assert result.north_m == 0.0
    # The uniform frame cannot form a validated bridge to the next textured frame.
    next_result = odom.process(_frame(_translate(terrain, 2), 1_400_000))
    assert not next_result.accepted
    assert next_result.north_m == 0.0
    assert next_result.segment_id == result.segment_id + 1


def test_out_of_order_sample_is_rejected_without_continuing_previous_path():
    terrain = _terrain()
    odom = ContinuousPlanarOdometry(_calibration())
    odom.process(_frame(terrain, 1_000_000))
    result = odom.process(_frame(terrain, 900_000))
    assert result.status == "NON_MONOTONIC_TIME"
    assert not result.accepted and result.north_m == 0.0
    init = odom.process(_frame(terrain, 1_200_000))
    assert init.status == "INITIALIZED" and init.segment_id == result.segment_id


def test_height_jump_restarts_segment():
    terrain = _terrain()
    odom = ContinuousPlanarOdometry(_calibration())
    odom.process(_frame(terrain, 1_000_000))
    result = odom.process(_frame(terrain, 1_200_000, h=120.0))
    assert result.status == "HEIGHT_JUMP"
    assert not result.accepted


def test_yaw_wrap_is_not_interpreted_as_full_turn():
    terrain = _terrain()
    odom = ContinuousPlanarOdometry(_calibration())
    odom.process(_frame(terrain, 1_000_000, yaw=pi - 0.03))
    result = odom.process(_frame(_translate(terrain, 2), 1_200_000,
                                     yaw=-pi + 0.03))
    assert result.accepted, result
    assert result.north_m == pytest.approx(2.0 * 100.0 / 1400.0, abs=0.06)


def test_invalid_frame_resets_and_requires_new_anchor():
    terrain = _terrain()
    odom = ContinuousPlanarOdometry(_calibration())
    odom.process(_frame(terrain, 1_000_000))
    broken = odom.process(_frame(terrain, 1_200_000, h=float("nan")))
    assert broken.status == "INVALID_FRAME"
    assert not broken.accepted
    restart = odom.process(_frame(terrain, 1_300_000))
    assert restart.status == "INITIALIZED"


def test_source_buffers_are_copied_on_ingest():
    image = _terrain()
    odom = ContinuousPlanarOdometry(_calibration())
    odom.process(_frame(image, 1_000_000))
    reference = image.copy()
    image[:, :] = 0
    result = odom.process(_frame(_translate(reference, 2), 1_200_000))
    assert result.accepted


def test_large_yaw_jump_cannot_produce_false_displacement():
    terrain = _terrain()
    odom = ContinuousPlanarOdometry(_calibration())
    odom.process(_frame(terrain, 1_000_000))
    bad = odom.process(_frame(_translate(terrain, 2), 1_200_000, yaw=0.6))
    assert not bad.accepted and bad.status == "EXCESSIVE_YAW_CHANGE"


def test_pitch_change_that_mimics_translation_is_rejected():
    terrain = _terrain()
    odom = ContinuousPlanarOdometry(_calibration())
    odom.process(_frame(terrain, 1_000_000, pitch=0.001))
    shifted = _translate(terrain, 3.0)
    out = odom.process(_frame(shifted, 1_200_000, pitch=0.015))
    assert out.status == "UNCOMPENSATED_TILT_CHANGE"
    assert not out.accepted
    assert out.north_m == 0.0


def test_static_frames_do_not_generate_false_horizontal_displacement():
    terrain = _terrain()
    odom = ContinuousPlanarOdometry(_calibration())
    odom.process(_frame(terrain, 1_000_000))
    for k in range(1, 5):
        out = odom.process(_frame(terrain, 1_000_000 + 100_000 * k))
        assert out.accepted, out
        assert abs(out.north_m) < 0.015
        assert abs(out.east_m) < 0.015
