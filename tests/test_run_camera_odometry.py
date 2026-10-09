import csv
import json

import cv2
import numpy as np
import pytest

from src.run_camera_odometry import run_manifest


def _frames():
    rng = np.random.default_rng(514)
    frame = np.zeros((480, 640), np.uint8)
    for _ in range(450):
        x, y = rng.integers([20, 20], [620, 460])
        cv2.circle(frame, (int(x), int(y)), 2, 190, -1)
    return frame


def _dataset(tmp_path, *, include_reference=True):
    base = _frames()
    fields = ["timestamp_us", "image_path", "height_agl_m",
              "roll_rad", "pitch_rad", "yaw_rad"]
    if include_reference:
        fields += ["true_n_m", "true_e_m"]
    samples = []
    for idx, dx in enumerate((0, 2, 4)):
        path = tmp_path / f"frame{idx}.png"
        shifted = cv2.warpAffine(base, np.float32([[1, 0, dx], [0, 1, 0]]),
                                 (640, 480))
        assert cv2.imwrite(str(path), shifted)
        record = {
            "timestamp_us": 1_000_000 + idx * 200_000,
            "image_path": path.name, "height_agl_m": 100,
            "roll_rad": 0, "pitch_rad": 0, "yaw_rad": 0,
        }
        if include_reference:
            record["true_n_m"] = -dx * 100 / 1400
            record["true_e_m"] = 0.0
        samples.append(record)

    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(samples)
    calibration = tmp_path / "calibration.yaml"
    calibration.write_text(
        "focal_x_px: 1400\nfocal_y_px: 1400\n"
        "center_x_px: 320\ncenter_y_px: 240\n"
        "image_to_body_xy: [[1, 0], [0, 1]]\n", encoding="utf-8"
    )
    return manifest, calibration


def test_manifest_runner_reports_segment_relative_error(tmp_path):
    manifest, calibration = _dataset(tmp_path)
    output = tmp_path / "out"
    report = run_manifest(manifest, calibration, output)
    assert report["frames"] == 3
    assert report["accepted_intervals"] == 2
    assert report["reference_available"]
    assert report["reference_evaluated_intervals"] == 2
    assert report["relative_position_rmse_m"] < 0.08
    assert report["relative_position_p95_m"] < 0.10
    assert (output / "visual_odometry.csv").is_file()
    assert json.loads((output / "visual_odometry_summary.json").read_text())["frames"] == 3


def test_manifest_runner_without_reference_never_claims_rmse(tmp_path):
    manifest, calibration = _dataset(tmp_path, include_reference=False)
    report = run_manifest(manifest, calibration, tmp_path / "out")
    assert report["frames"] == 3
    assert report["relative_position_rmse_m"] is None
    assert report["reference_evaluated_intervals"] == 0


def test_manifest_runner_rejects_missing_image(tmp_path):
    manifest, calibration = _dataset(tmp_path)
    (tmp_path / "frame1.png").unlink()
    with pytest.raises(ValueError, match="не удалось прочитать кадр"):
        run_manifest(manifest, calibration, tmp_path / "out")


def test_reference_error_never_connects_across_unobserved_time_gap(tmp_path):
    manifest, calibration = _dataset(tmp_path)
    with manifest.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames
        records = list(reader)

    # A discontinuous reference trajectory across a missing time interval.
    # The estimator must re-anchor truth independently for the next segment.
    records[2]["timestamp_us"] = 2_000_000
    records[2]["true_n_m"] = 1000.0
    records[2]["true_e_m"] = -500.0
    records.append({
        **records[2],
        "timestamp_us": 2_200_000,
        "image_path": "frame3.png",
        "true_n_m": 1000.0 - 2.0 * 100.0 / 1400.0,
        "true_e_m": -500.0,
    })
    # The final image is shifted only two pixels relative to the new anchor.
    from_frame = cv2.imread(str(tmp_path / "frame2.png"), cv2.IMREAD_GRAYSCALE)
    img = cv2.warpAffine(from_frame, np.float32([[1, 0, 2], [0, 1, 0]]),
                         (640, 480))
    assert cv2.imwrite(str(tmp_path / "frame3.png"), img)
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    report = run_manifest(manifest, calibration, tmp_path / "out")
    assert report["independent_segments"] == 2
    assert report["accepted_intervals"] == 2
    assert report["reference_evaluated_intervals"] == 2
    assert report["relative_position_rmse_m"] < 0.10
