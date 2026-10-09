"""Сквозной тест запаздывающей визуальной скорости по двум журналам."""
import csv
import json
import pytest

from src.run_delayed_eskf_replay import run_delayed_replay


def make_logs(tmp_path):
    settings = tmp_path / "settings.yaml"
    settings.write_text(
        "stationary_calibration_confirmed: true\n"
        "initial_quaternion_body_to_ned: [1,0,0,0]\n"
        "gyro_bias_body_rad_s: [0,0,0]\n"
        "accel_bias_body_m_s2: [0,0,0]\n"
        "gyro_rad_s_sqrt_hz: 0.003\n"
        "accel_m_s2_sqrt_hz: 0.03\n"
        "gyro_bias_walk_rad_s2_sqrt_hz: 0.00005\n"
        "accel_bias_walk_m_s3_sqrt_hz: 0.0003\n"
        "max_visual_lag_us: 120000\n"
        "max_buffered_imu_samples: 50\n"
        "allow_experimental_visual_covariance: true\n",
        encoding="utf-8",
    )
    imu = tmp_path / "imu.csv"
    with imu.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "timestamp_us", "gyro_x_rad_s", "gyro_y_rad_s", "gyro_z_rad_s",
            "accel_x_m_s2", "accel_y_m_s2", "accel_z_m_s2",
        ])
        for i in range(41):
            w.writerow([1_000_000+i*10000, 0,0,0, 0,0,-9.80665])
    visual = tmp_path / "visual.csv"
    with visual.open("w", newline="", encoding="utf-8") as f:
        w=csv.writer(f)
        w.writerow([
            "timestamp_us", "arrival_timestamp_imu_us", "accepted",
            "vn_m_s", "ve_m_s", "velocity_cov_nn_m2_s2",
            "velocity_cov_ne_m2_s2", "velocity_cov_ee_m2_s2",
        ])
        # Measurements arrive out of the exposure-time order.
        w.writerow([1_105_000, 1_195_000, 1, 0,0, 0.004,0,0.004])
        w.writerow([1_155_000, 1_180_000, 1, 0,0, 0.004,0,0.004])
        w.writerow([1_205_000, 1_350_000, 1, 0,0, 0.004,0,0.004])
    return imu,visual,settings


def test_delayed_runner_reorders_and_declines_overdue_visuals(tmp_path):
    imu,visual,settings = make_logs(tmp_path)
    result=run_delayed_replay(imu,visual,settings,tmp_path/"out")
    assert result["raw_imu_samples"]==41
    assert result["visual_samples"]==3
    assert result["accepted_visual_updates"]==2
    assert result["statuses"]["VISUAL_CORRECTED"]==2
    assert result["statuses"]["VISUAL_DELAY_EXCEEDED"]==1
    with (tmp_path/"out"/"delayed_visual_replay.csv").open(newline="") as f:
        rows=list(csv.DictReader(f))
    assert [int(r["exposure_timestamp_imu_us"]) for r in rows] == [
        1_155_000,1_105_000,1_205_000
    ]
    assert all(abs(float(row["vn_m_s"]))<1e-8 for row in rows)
    assert json.loads((tmp_path/"out"/"delayed_visual_summary.json").read_text())[
        "accepted_visual_updates"]==2


def test_arrival_before_exposure_is_rejected(tmp_path):
    imu, visual, settings = make_logs(tmp_path)
    with visual.open(newline="") as f:
        rows=list(csv.reader(f))
    rows[1][1]="1"
    with visual.open("w", newline="") as f:
        csv.writer(f).writerows(rows)
    with pytest.raises(ValueError,match="раньше экспозиции"):
        run_delayed_replay(imu,visual,settings,tmp_path/"out")
