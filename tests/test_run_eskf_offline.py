"""Сквозные проверки исследовательского 15-мерного фильтра по CSV."""
import csv
import json

import pytest

from src.run_eskf_offline import run_eskf_offline


def _inputs(tmp_path, *, experimental=True):
    yaml_path = tmp_path / "noise.yaml"
    yaml_path.write_text(
        "stationary_calibration_confirmed: true\n"
        "initial_quaternion_body_to_ned: [1, 0, 0, 0]\n"
        "gyro_bias_body_rad_s: [0, 0, 0]\n"
        "accel_bias_body_m_s2: [0, 0, 0]\n"
        "gyro_rad_s_sqrt_hz: 0.003\n"
        "accel_m_s2_sqrt_hz: 0.03\n"
        "gyro_bias_walk_rad_s2_sqrt_hz: 0.00005\n"
        "accel_bias_walk_m_s3_sqrt_hz: 0.0003\n"
        f"allow_experimental_visual_covariance: {str(experimental).lower()}\n",
        encoding="utf-8",
    )
    imu_path = tmp_path / "imu.csv"
    with imu_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "timestamp_us", "gyro_x_rad_s", "gyro_y_rad_s", "gyro_z_rad_s",
            "accel_x_m_s2", "accel_y_m_s2", "accel_z_m_s2",
        ])
        for k in range(31):
            writer.writerow([1_000_000+k*10_000, 0, 0, 0, 0, 0, -9.80665])
    visual_path = tmp_path / "visual.csv"
    with visual_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "timestamp_us", "accepted", "vn_m_s", "ve_m_s",
            "velocity_cov_nn_m2_s2", "velocity_cov_ne_m2_s2",
            "velocity_cov_ee_m2_s2",
        ])
        for time_us in (1_005_000, 1_105_000, 1_205_000):
            writer.writerow([time_us, 1, 0, 0, 0.003, 0, 0.003])
    return imu_path, visual_path, yaml_path


def test_stationary_imu_camera_log_does_not_create_horizontal_motion(tmp_path):
    imu, visual, config = _inputs(tmp_path)
    out = run_eskf_offline(imu, visual, config, tmp_path / "out")
    assert out["visual_samples"] == 3
    assert out["accepted_visual_updates"] == 3
    assert out["reasons"] == {"VISUAL_CORRECTED": 3}
    assert out["nis_mean"] == pytest.approx(0., abs=1e-9)
    assert out["covariance_mode"] == "EXPERIMENTAL_UNVALIDATED"
    with (tmp_path / "out" / "eskf_trajectory.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    assert all(abs(float(row["vn_m_s"])) < 1e-9 for row in rows)
    with (tmp_path / "out" / "eskf_states.jsonl").open(encoding="utf-8") as f:
        snapshots = [json.loads(line) for line in f if line.strip()]
    assert len(snapshots) == 3
    assert all(len(row["covariance_15x15"]) == 15 for row in snapshots)
    assert all(len(row["covariance_15x15"][0]) == 15 for row in snapshots)
    assert [s["timestamp_us"] for s in snapshots] == [
        1_005_000, 1_105_000, 1_205_000
    ]
    assert json.loads((tmp_path / "out" / "eskf_summary.json").read_text())[
        "accepted_visual_updates"
    ] == 3


def test_rejects_implicit_unverified_covariance(tmp_path):
    imu, visual, config = _inputs(tmp_path, experimental=False)
    with pytest.raises(ValueError, match="ковариация"):
        run_eskf_offline(imu, visual, config, tmp_path / "out")


def test_high_speed_outlier_is_recorded_not_fused(tmp_path):
    imu, visual, config = _inputs(tmp_path)
    with visual.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    rows[2][2] = "200"
    with visual.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    summary = run_eskf_offline(imu, visual, config, tmp_path / "out")
    assert summary["reasons"]["VISUAL_OUTLIER_REJECTED"] == 1
    assert summary["accepted_visual_updates"] == 2
    assert summary["nis_mean"] > 9.21


def test_large_gap_in_imu_prevents_unsupported_visual_update(tmp_path):
    imu, visual, config = _inputs(tmp_path)
    with imu.open(newline="", encoding="utf-8") as stream:
        data = list(csv.reader(stream))
    data = [data[0]] + [
        row for row in data[1:]
        if int(row[0]) <= 1_060_000 or int(row[0]) >= 1_160_000
    ]
    with imu.open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerows(data)
    report = run_eskf_offline(imu, visual, config, tmp_path / "out")
    assert report["accepted_visual_updates"] <= 1
    assert (report["reasons"].get("IMU_INVALID_OR_GAP", 0) +
            report["reasons"].get("IMU_GAP_AT_CAMERA_FRAME", 0)) >= 1
