"""Сквозная проверка офлайн-слияния журналов ИИМ/визуальной одометрии."""
import csv
import json

import pytest

from src.run_imu_visual_fusion import run_fusion


def make_inputs(tmp_path, *, stationary_confirmed=True):
    config = tmp_path / "calibration.yaml"
    config.write_text(
        f"stationary_calibration_confirmed: {str(stationary_confirmed).lower()}\n"
        "initial_quaternion_body_to_ned: [1, 0, 0, 0]\n"
        "gyro_bias_body_rad_s: [0, 0, 0]\n"
        "accel_bias_body_m_s2: [0, 0, 0]\n"
        "gyro_noise_rad_s: 0.004\n"
        "accel_noise_m_s2: 0.08\n"
        "accel_bias_walk_m_s2_sqrt_s: 0.002\n"
        "max_imu_dt_s: 0.05\n",
        encoding="utf-8"
    )
    imu = tmp_path / "imu.csv"
    with imu.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp_us", "gyro_x_rad_s", "gyro_y_rad_s", "gyro_z_rad_s",
            "accel_x_m_s2", "accel_y_m_s2", "accel_z_m_s2",
        ])
        for i in range(21):
            writer.writerow([
                1_000_000 + 10_000 * i, 0, 0, 0, 0, 0, -9.80665
            ])
    visual = tmp_path / "visual.csv"
    with visual.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp_us", "accepted", "vn_m_s", "ve_m_s",
            "velocity_cov_nn_m2_s2", "velocity_cov_ne_m2_s2",
            "velocity_cov_ee_m2_s2"
        ])
        for t in (1_005_000, 1_055_000, 1_105_000):
            writer.writerow([t, 1, 0, 0, 0.0025, 0, 0.0025])
    return imu, visual, config


def test_fusion_runner_interpolates_imu_to_exposure_time(tmp_path):
    imu, visual, cfg = make_inputs(tmp_path)
    report = run_fusion(imu, visual, cfg, tmp_path / "out")
    assert report["accepted_visual_updates"] == 3
    assert report["status_counts"] == {"VISUAL_CORRECTED": 3}
    assert report["imu_samples"] == 21
    with (tmp_path / "out" / "imu_visual_fusion.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    assert all(abs(float(r["vn_m_s"])) < 1e-10 for r in rows)
    assert all(abs(float(r["relative_north_m"])) < 1e-10 for r in rows)
    assert json.loads((tmp_path / "out" / "imu_visual_summary.json").read_text())[
        "accepted_visual_updates"
    ] == 3


def test_fusion_runner_requires_confirmed_stationary_bias_calibration(tmp_path):
    imu, visual, cfg = make_inputs(tmp_path, stationary_confirmed=False)
    with pytest.raises(ValueError, match="подтвержденная"):
        run_fusion(imu, visual, cfg, tmp_path / "out")


def test_visual_without_covariance_never_fuses(tmp_path):
    imu, visual, cfg = make_inputs(tmp_path)
    with visual.open(newline="") as f:
        data = list(csv.reader(f))
    data[1][4] = "nan"
    with visual.open("w", newline="") as f:
        csv.writer(f).writerows(data)
    result = run_fusion(imu, visual, cfg, tmp_path / "out")
    assert result["accepted_visual_updates"] == 2
    assert result["status_counts"]["VISUAL_COVARIANCE_INVALID"] == 1


def test_duplicate_imu_time_is_refused(tmp_path):
    imu, visual, cfg = make_inputs(tmp_path)
    with imu.open(newline="") as f:
        data = list(csv.reader(f))
    data[2][0] = data[1][0]
    with imu.open("w", newline="") as f:
        csv.writer(f).writerows(data)
    with pytest.raises(ValueError, match="время не возрастает"):
        run_fusion(imu, visual, cfg, tmp_path / "out")


def test_gap_prevents_visual_update_without_inventing_imu_samples(tmp_path):
    imu, visual, cfg = make_inputs(tmp_path)
    with imu.open(newline="") as f:
        data = list(csv.reader(f))
    data = [data[0]] + [
        row for row in data[1:]
        if int(row[0]) <= 1_030_000 or int(row[0]) >= 1_130_000
    ]
    with imu.open("w", newline="") as f:
        csv.writer(f).writerows(data)
    result = run_fusion(imu, visual, cfg, tmp_path / "out")
    assert result["accepted_visual_updates"] <= 1
    assert result["status_counts"]["IMU_GAP_AT_FRAME"] >= 1
