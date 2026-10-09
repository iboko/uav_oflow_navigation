"""Математические проверки начальной модели неопределенности скорости."""
from math import isfinite
import csv

import cv2
import numpy as np
import pytest

from src.ground_visual_motion import CameraCalibration, GroundMotionEstimate
from src.velocity_uncertainty import (
    VelocityErrorAssumptions, modeled_velocity_covariance_ned
)
from src.run_camera_odometry import run_manifest


def _camera():
    return CameraCalibration(1000., 1000., 320., 240.,
                             ((1., 0.), (0., 1.)))


def _motion():
    return GroundMotionEstimate(True, "OK", 0.2, 0.1, 2., -1., 100, 80, 0.8)


def _assumptions(**updates):
    kwargs = dict(
        fitted_displacement_sigma_px=0.7,
        height_sigma_m=0.2,
        interval_sigma_s=0.001,
        yaw_sigma_rad=0.005,
        focal_relative_sigma=0.005,
        camera_offset_sigma_m=0.03,
        velocity_model_floor_m_s=0.01,
    )
    kwargs.update(updates)
    return VelocityErrorAssumptions(**kwargs)


def _cov(assumptions, height=100., interval=0.2, rpy_cur=(0., 0., 0.)):
    return modeled_velocity_covariance_ned(
        assumptions, camera=_camera(), motion=_motion(),
        height_agl_m=height, interval_s=interval,
        previous_rpy=(0., 0., 0.), current_rpy=rpy_cur,
        center_velocity_ned_m_s=(0.2, 0.1),
    )


def test_covariance_symmetric_positive_definite():
    covariance = _cov(_assumptions())
    assert covariance.shape == (2, 2)
    assert np.allclose(covariance, covariance.T, atol=1e-14)
    assert np.linalg.eigvalsh(covariance).min() > 0.


def test_more_distance_or_less_time_increases_pixel_velocity_uncertainty():
    model = _assumptions()
    trace_near = np.trace(_cov(model, height=20.0))
    trace_far = np.trace(_cov(model, height=100.0))
    assert trace_far > trace_near
    assert np.trace(_cov(model, interval=0.1)) > np.trace(_cov(model, interval=0.2))


def test_large_height_and_camera_lever_uncertainty_inflate_covariance():
    standard = _cov(_assumptions(), rpy_cur=(0., 0., 0.2))
    lever = _cov(_assumptions(camera_offset_sigma_m=0.4),
                 rpy_cur=(0., 0., 0.2))
    assert np.trace(lever) > np.trace(standard)
    height_error = _cov(_assumptions(height_sigma_m=5.0))
    assert np.trace(height_error) > np.trace(_cov(_assumptions()))


def test_uncertainty_requires_explicit_coefficient_values():
    assert not _assumptions(fitted_displacement_sigma_px=0).valid()
    assert not _assumptions(velocity_model_floor_m_s=0).valid()
    with pytest.raises(ValueError):
        _cov(_assumptions(yaw_sigma_rad=float("nan")))


def test_log_records_heuristic_covariance_only_when_configured(tmp_path):
    rng = np.random.default_rng(76)
    img = np.zeros((480, 640), np.uint8)
    for _ in range(430):
        x, y = rng.integers([20, 20], [620, 460])
        cv2.circle(img, (int(x), int(y)), 2, 200, -1)
    calibration = tmp_path / "camera.yaml"
    base = (
        "focal_x_px: 1000\nfocal_y_px: 1000\n"
        "center_x_px: 320\ncenter_y_px: 240\n"
        "image_to_body_xy: [[1, 0], [0, 1]]\n"
    )
    calibration.write_text(base, encoding="utf-8")
    manifest = tmp_path / "frames.csv"
    with manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_us", "image_path", "height_agl_m",
                         "roll_rad", "pitch_rad", "yaw_rad"])
        for i, dx in enumerate([0, 2]):
            name = f"{i}.png"
            assert cv2.imwrite(str(tmp_path / name),
                cv2.warpAffine(img, np.float32([[1, 0, dx], [0, 1, 0]]),
                                  (640, 480)))
            writer.writerow([1_000_000 + 200_000 * i, name, 100, 0, 0, 0])
    without = run_manifest(manifest, calibration, tmp_path / "without")
    assert without["velocity_covariance_source"] == "NOT_ESTIMATED"
    calibration.write_text(base + (
        "velocity_error_assumptions:\n"
        "  fitted_displacement_sigma_px: 0.7\n"
        "  height_sigma_m: 0.2\n"
        "  interval_sigma_s: 0.001\n"
        "  yaw_sigma_rad: 0.005\n"
        "  focal_relative_sigma: 0.005\n"
        "  velocity_model_floor_m_s: 0.01\n"
    ), encoding="utf-8")
    with_priors = run_manifest(manifest, calibration, tmp_path / "with")
    assert with_priors["velocity_covariance_source"] == "ENGINEERING_ASSUMPTIONS_NOT_VALIDATED"
    with (tmp_path / "with" / "visual_odometry.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert isfinite(float(rows[1]["velocity_cov_nn_m2_s2"]))
    assert float(rows[1]["velocity_cov_nn_m2_s2"]) > 0.
    assert rows[0]["accepted"] == "0"
