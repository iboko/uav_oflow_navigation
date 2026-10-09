"""Геометрические тесты компенсации вращения нижней камеры."""
import numpy as np
import cv2
import pytest

from src.camera_rotation_compensation import (
    CameraMountCalibration, align_current_to_previous,
    body_to_ned, rotation_homography_previous_to_current,
)
from src.continuous_visual_odometry import ContinuousPlanarOdometry, VisualFrame
from src.ground_visual_motion import CameraCalibration

CAMERA_TO_BODY = (
    (0.0, -1.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0),
)


def _camera():
    return CameraCalibration(900.0, 900.0, 320.0, 240.0,
                             ((0.0, -1.0), (1.0, 0.0)))


def _mount():
    return CameraMountCalibration(CAMERA_TO_BODY)


def _terrain():
    rng = np.random.default_rng(56)
    img = np.zeros((480, 640), np.uint8)
    for _ in range(700):
        x, y = rng.integers([25, 25], [615, 455])
        cv2.circle(img, (int(x), int(y)), 2, int(rng.integers(120, 250)), -1)
    return img


def _warp_rotation(image, *, pitch=0.0, roll=0.0, yaw=0.0):
    h = rotation_homography_previous_to_current(
        _camera(), _mount(), previous_rpy=(0.0, 0.0, 0.0),
        current_rpy=(roll, pitch, yaw)
    )
    return cv2.warpPerspective(image, h, (640, 480))


def test_mount_matrix_is_righthanded_and_compatible_with_body_axes():
    assert _mount().valid(_camera())
    other = CameraCalibration(900, 900, 320, 240, ((1, 0), (0, 1)))
    assert not _mount().valid(other)
    reflected = CameraMountCalibration(((0, 1, 0), (1, 0, 0), (0, 0, 1)))
    assert not reflected.valid(_camera())


def test_body_to_ned_is_proper_rotation():
    r = body_to_ned(0.12, -0.06, 0.32)
    assert np.allclose(r.T @ r, np.eye(3), atol=1e-12)
    assert np.linalg.det(r) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("roll,pitch,yaw", [
    (0.012, 0.0, 0.0),
    (0.0, 0.012, 0.0),
    (0.0, 0.0, 0.06),
    (0.012, -0.010, 0.045),
])
def test_pure_rotation_does_not_fabricate_horizontal_translation(roll, pitch, yaw):
    base = _terrain()
    warped = _warp_rotation(base, roll=roll, pitch=pitch, yaw=yaw)
    odom = ContinuousPlanarOdometry(_camera(), camera_mount=_mount())
    start = odom.process(VisualFrame(1_000_000, base, 100.0, 0.0, 0.0, 0.0))
    end = odom.process(VisualFrame(1_200_000, warped, 100.0, yaw, roll, pitch))
    assert start.status == "INITIALIZED"
    assert end.accepted, end
    assert abs(end.north_m) < 0.10
    assert abs(end.east_m) < 0.10


def test_translation_plus_pitch_is_preserved_after_angular_compensation():
    base = _terrain()
    warped_translation = cv2.warpAffine(
        base, np.float32([[1, 0, 3], [0, 1, 0]]), (640, 480)
    )
    pitched = _warp_rotation(warped_translation, pitch=0.014)
    odom = ContinuousPlanarOdometry(_camera(), camera_mount=_mount())
    odom.process(VisualFrame(1_000_000, base, 100.0, 0.0))
    result = odom.process(VisualFrame(1_200_000, pitched, 100.0, 0.0, pitch_rad=0.014))
    assert result.accepted, result
    # Positive image-X shift maps to negative body-right displacement.
    assert result.north_m == pytest.approx(0.0, abs=0.09)
    assert result.east_m == pytest.approx(-3 * 100 / 900, abs=0.10)


def test_no_3d_mount_preserves_original_rejection_of_pitch_changes():
    image = _terrain()
    odom = ContinuousPlanarOdometry(_camera())
    odom.process(VisualFrame(1_000_000, image, 100.0, 0.0))
    end = odom.process(VisualFrame(
        1_200_000, _warp_rotation(image, pitch=0.014), 100.0, 0.0,
        pitch_rad=0.014
    ))
    assert end.status == "UNCOMPENSATED_TILT_CHANGE"
    assert not end.accepted


def test_bad_mount_or_insufficient_overlap_fails_closed():
    base = _terrain()
    with pytest.raises(ValueError):
        ContinuousPlanarOdometry(
            _camera(), camera_mount=CameraMountCalibration(((1., 0., 0.),
                                                            (0., 1., 0.),
                                                            (0., 0., 1.)))
        )
    with pytest.raises(ValueError, match="перекрытие"):
        align_current_to_previous(
            base, _camera(), _mount(),
            previous_rpy=(0.0, 0.0, 0.0),
            current_rpy=(0.0, 0.12, 0.0),
            min_overlap_fraction=0.98
        )
