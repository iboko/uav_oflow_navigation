"""Калибровка ошибок карты/VO стенда и слабосвязанная VIO с картой."""
import csv
import json

import pytest

from src.export_px4_sitl_replay import export_eskf_events
from src.synthetic_bench_calibration import calibrate_bench
from src.synthetic_sensor_bench import (
    BenchConfig, generate_dataset, load_bench_calibration, run_sensor_bench,
)


@pytest.fixture(scope="module")
def calibration(tmp_path_factory):
    work = tmp_path_factory.mktemp("cal")
    data = calibrate_bench([9101, 9102, 9103], [9201, 9202], frames=16,
                           work_dir=work, min_samples_per_flight=12)
    path = work / "calibration.json"
    path.write_text(json.dumps(data))
    return path, data


def test_calibration_is_from_disjoint_flights_and_holdout_consistent(calibration):
    _, data = calibration
    assert data["synthetic_only"] and not data["independently_validated_for_flight"]
    assert not set(data["seeds"]["fit"]) & set(data["seeds"]["holdout"])
    assert data["vo_body_velocity"]["consistent_on_holdout"]
    assert 1.0 < data["map_ground_point"]["holdout_mean_nees_2d"] < 3.5
    assert 0.3 < data["map_yaw"]["holdout_mean_nees_1d"] < 2.5


def test_evaluation_seed_used_for_calibration_is_refused(calibration):
    path, _ = calibration
    with pytest.raises(ValueError):
        load_bench_calibration(path, evaluation_seed=9101)


def test_vio_with_calibrated_ground_point_and_heading(tmp_path, calibration):
    path, _ = calibration
    folder = generate_dataset(tmp_path / "in", BenchConfig(frames=10, seed=72118))
    summary = run_sensor_bench(folder, tmp_path / "out",
                               calibration=load_bench_calibration(path, evaluation_seed=72118))
    assert summary["attitude_source"] == "ESKF_INTERNAL"
    assert summary["map_position_updates_fused"] >= 7
    assert summary["map_yaw_updates_fused"] >= 7
    assert summary["applied_map_correction"]
    assert summary["eskf_position_rmse_m"] < .2
    assert summary["eskf_speed_rmse_m_s"] < .1
    assert summary["eskf_yaw_error_max_rad"] < .01
    with (tmp_path / "out" / "sensor_bench_results.csv").open(newline="") as fp:
        rows = list(csv.DictReader(fp))
    # Body velocity is stamped at the midpoint of the frame interval.
    timed = [r for r in rows if r["body_vx_vo_m_s"] not in ("", "nan")]
    assert all(int(r["body_time_us"]) < int(r["timestamp_us"]) for r in timed)


def test_ambiguous_map_is_never_fused_even_when_calibrated(tmp_path, calibration):
    path, _ = calibration
    folder = generate_dataset(tmp_path / "in", BenchConfig(
        scenario="duplicate_map", frames=6, seed=72118))
    summary = run_sensor_bench(folder, tmp_path / "out",
                               calibration=load_bench_calibration(path, evaluation_seed=72118))
    assert summary["map_position_updates_fused"] == 0
    assert summary["position_reference"] == "RELATIVE_TO_START"
    assert summary["visual_updates_fused"] >= 3


def test_eskf_event_stream_for_px4_is_arrival_ordered(tmp_path):
    folder = generate_dataset(tmp_path / "in", BenchConfig(frames=6, seed=72118))
    run_sensor_bench(folder, tmp_path / "out")
    expected = export_eskf_events(tmp_path / "out" / "sensor_bench_results.csv",
                                  folder / "imu.csv", tmp_path / "events.csv",
                                  tmp_path / "expected.json")
    with (tmp_path / "events.csv").open(newline="") as fp:
        rows = list(csv.DictReader(fp))
    arrivals = [int(r["arrival_us"]) for r in rows]
    assert arrivals == sorted(arrivals)
    assert expected["body_velocity_events"] == sum(r["kind"] == "B" for r in rows) >= 4
    assert expected["statuses"].get("VISUAL_CORRECTED", 0) >= 4
    assert expected["publish_to_ekf2"] is False
    assert all(int(r["arrival_us"]) > int(r["time_us"]) for r in rows if r["kind"] == "B")
