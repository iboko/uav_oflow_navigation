import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from src.ground_visual_motion import CameraCalibration, estimate_ground_motion


def calibration():
    # Identity is a synthetic test fixture, not a flight installation default.
    return CameraCalibration(
        focal_x_px=1400.0,
        focal_y_px=1400.0,
        center_x_px=320.0,
        center_y_px=240.0,
        image_to_body_xy=((1.0, 0.0), (0.0, 1.0)),
    )


def textured_frame():
    rng = np.random.default_rng(14)
    image = np.zeros((480, 640), np.uint8)
    for _ in range(350):
        x, y = rng.integers([18, 18], [622, 462])
        cv2.circle(image, (int(x), int(y)), 2, int(rng.integers(100, 256)), -1)
    return image


def test_nadir_camera_translation_100m():
    image = textured_frame()
    dx_px, dy_px = 3.0, -2.0
    warped = cv2.warpAffine(
        image, np.float32([[1, 0, dx_px], [0, 1, dy_px]]),
        (640, 480)
    )
    out = estimate_ground_motion(
        image, warped, dt_s=0.2, height_agl_m=100.0,
        calibration=calibration()
    )
    assert out.valid, out
    assert out.tracked_points >= 20
    assert out.inlier_ratio >= 0.6
    assert out.velocity_body_x_m_s == pytest.approx(
        -dx_px * 100.0 / (1400.0 * 0.2), abs=0.15
    )
    assert out.velocity_body_y_m_s == pytest.approx(
        -dy_px * 100.0 / (1400.0 * 0.2), abs=0.15
    )


def test_textureless_image_is_rejected():
    image = np.full((480, 640), 127, dtype=np.uint8)
    out = estimate_ground_motion(image, image, 0.2, 100.0, calibration())
    assert not out.valid
    assert out.reason == "INSUFFICIENT_TEXTURE"


def test_excessive_tilt_rejected_before_tracking():
    image = textured_frame()
    out = estimate_ground_motion(
        image, image, 0.2, 100.0, calibration(), roll_rad=0.4
    )
    assert not out.valid
    assert out.reason == "TILT_EXCEEDS_PLANAR_MODEL"


def test_invalid_axis_calibration_is_rejected():
    image = textured_frame()
    bad = CameraCalibration(
        1400.0, 1400.0, 320.0, 240.0,
        ((2.0, 0.0), (0.0, 1.0))
    )
    out = estimate_ground_motion(image, image, 0.2, 100.0, bad)
    assert not out.valid
    assert out.reason == "INVALID_CALIBRATION"
