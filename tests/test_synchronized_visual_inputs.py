"""Тесты учета плеча установки камеры и сквозной обработки синхронных логов."""
import csv
from math import cos, sin
import cv2
import numpy as np
import pytest

from src.camera_rotation_compensation import (
    CameraMountCalibration, camera_delta_to_center_ned,
    rotation_homography_previous_to_current,
)
from src.continuous_visual_odometry import ContinuousPlanarOdometry, VisualFrame
from src.ground_visual_motion import CameraCalibration
from src.run_camera_odometry import run_manifest


def calib():
    return CameraCalibration(900., 900., 320., 240.,
                             ((0., -1.), (1., 0.)))


def mount():
    return CameraMountCalibration(((0., -1., 0.),
                                   (1., 0., 0.),
                                   (0., 0., 1.)))


def terrain():
    image = np.zeros((480, 640), dtype=np.uint8)
    rng = np.random.default_rng(211)
    for _ in range(700):
        x, y = rng.integers([20, 20], [620, 460])
        cv2.circle(image, (int(x), int(y)), 2, 225, -1)
    return image


def test_analytical_camera_lever_arm_pure_rotation_yields_zero_center_motion():
    yaw = 0.12
    r = np.array([2., 0., 0.])
    r_new = np.array([2. * cos(yaw), 2. * sin(yaw), 0.])
    delta_cam = r_new - r
    result = camera_delta_to_center_ned(
        tuple(delta_cam[:2]), (0., 0., 0.), (0., 0., yaw),
        tuple(r),
    )
    assert np.allclose(result, [0., 0.], atol=1e-12)


def test_analytical_lever_correction_preserves_center_translation():
    center = np.array([0.7, -0.35])
    yaw = 0.13
    r = (1.2, -0.3, 0.15)
    rot0 = np.eye(3)
    rot1 = np.array([[cos(yaw), -sin(yaw), 0],
                     [sin(yaw), cos(yaw), 0],
                     [0, 0, 1]])
    camera_delta = center + ((rot1 - rot0) @ np.array(r))[:2]
    recovered = camera_delta_to_center_ned(
        tuple(camera_delta), (0., 0., 0.), (0., 0., yaw), r,
    )
    assert np.allclose(recovered, center, atol=1e-12)


def test_nonzero_lever_without_camera_mount_must_fail():
    with pytest.raises(ValueError, match="трехмерной"):
        ContinuousPlanarOdometry(calib(), camera_offset_body_m=(1., 0., 0.))


def test_zero_center_velocity_with_camera_orbit_synthetic_image():
    image = terrain()
    yaw = 0.12
    lever = np.array([2., 0., 0.])
    camera_delta = np.array([2. * (cos(yaw) - 1.), 2. * sin(yaw)])
    # camera_delta_body = R_image_to_body * (-height / focal) * pixel_shift
    shift_image = -900. / 100. * np.array([camera_delta[1], -camera_delta[0]])
    shifted = cv2.warpAffine(
        image, np.float32([[1, 0, shift_image[0]],
                           [0, 1, shift_image[1]]]), (640, 480))
    h = rotation_homography_previous_to_current(
        calib(), mount(), previous_rpy=(0., 0., 0.),
        current_rpy=(0., 0., yaw))
    second = cv2.warpPerspective(shifted, h, (640, 480))

    odom = ContinuousPlanarOdometry(
        calib(), camera_mount=mount(), camera_offset_body_m=tuple(lever)
    )
    assert odom.process(VisualFrame(1_000_000, image, 100., 0.)).status == "INITIALIZED"
    result = odom.process(VisualFrame(1_200_000, second, 100., yaw))
    assert result.accepted, result
    assert abs(result.north_m) < 0.12
    assert abs(result.east_m) < 0.12


def test_manifest_uses_attitude_csv_and_fails_closed_on_gaps(tmp_path):
    image = terrain()
    calibration = tmp_path / "camera.yaml"
    calibration.write_text(
        "focal_x_px: 900\nfocal_y_px: 900\n"
        "center_x_px: 320\ncenter_y_px: 240\n"
        "image_to_body_xy: [[0, -1], [1, 0]]\n"
        "body_from_camera: [[0, -1, 0], [1, 0, 0], [0, 0, 1]]\n"
        "camera_offset_body_m: [0, 0, 0]\n", encoding="utf-8")
    manifest = tmp_path / "frames.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["timestamp_us", "image_path", "height_agl_m"])
        writer.writeheader()
        for index, dx in enumerate([0, 2, 4]):
            file = tmp_path / f"{index}.png"
            shifted = cv2.warpAffine(
                image, np.float32([[1, 0, dx], [0, 1, 0]]), (640, 480)
            )
            assert cv2.imwrite(str(file), shifted)
            writer.writerow({
                "timestamp_us": 1_000_000 + index * 200_000,
                "image_path": file.name, "height_agl_m": "100",
            })
    timeline = tmp_path / "attitude.csv"
    def save_att(samples):
        with timeline.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp_us", "qw", "qx", "qy", "qz"])
            for t in samples:
                writer.writerow([t, 1., 0., 0., 0.])

    # Camera clock + 5000 us -> attitude clock. Both bracketing records exist.
    save_att([1_000_000, 1_010_000, 1_200_000, 1_210_000,
              1_400_000, 1_410_000])
    good = run_manifest(
        manifest, calibration, tmp_path / "good",
        attitude_log=timeline, camera_to_attitude_offset_us=5000,
        max_attitude_gap_us=20_000,
    )
    assert good["attitude_synchronized_from_log"]
    assert good["accepted_intervals"] == 2
    assert good["status_counts"]["INITIALIZED"] == 1

    # Remove the two records surrounding the second frame. Do not
    # extrapolate from remote samples and do not bridge the missing interval.
    save_att([1_000_000, 1_010_000, 1_400_000, 1_410_000])
    bad = run_manifest(
        manifest, calibration, tmp_path / "bad",
        attitude_log=timeline, camera_to_attitude_offset_us=5000,
        max_attitude_gap_us=20_000,
    )
    assert bad["status_counts"]["ATTITUDE_SYNC_FAILED"] == 1
    assert bad["accepted_intervals"] == 0
