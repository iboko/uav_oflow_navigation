"""Image/IMU/range consistency and end-to-end actual estimator regression."""
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.synthetic_sensor_bench import (
    BenchConfig, _calibration, _terrain, generate_dataset,
    render_camera_frame, run_sensor_bench,
)


def _run(tmp_path, scenario="nominal", frames=5):
    folder = generate_dataset(
        tmp_path / "input", BenchConfig(
            scenario=scenario, frames=frames, seed=72118
        )
    )
    result = run_sensor_bench(folder, tmp_path/"output")
    return folder, result


def test_planar_homography_is_physically_consistent_with_100m_map_crop():
    camera,mount,georef=_calibration()
    terrain=_terrain(72118)
    rendered=render_camera_frame(
        terrain, np.array([50.,6.]), 0., height_m=100.,
        camera=camera,mount=mount,georef=georef
    )
    expected=terrain[260:740,240:880]
    assert rendered.shape==(480,640)
    assert np.mean(np.abs(rendered.astype(float)-expected.astype(float)))<.25


def test_synthetic_sensor_manifest_has_one_time_basis_and_truth_is_separate(tmp_path):
    folder=generate_dataset(
        tmp_path, BenchConfig(frames=5, scenario="nominal")
    )
    meta=json.loads((folder/"bench_metadata.json").read_text())
    assert meta["synthetic_only"]
    assert not meta["ground_truth_used_by_estimators"]
    with (folder/"imu.csv").open(newline="") as f:
        imu=list(csv.DictReader(f))
    with (folder/"camera.csv").open(newline="") as f:
        camera=list(csv.DictReader(f))
    with (folder/"truth.csv").open(newline="") as f:
        truth=list(csv.DictReader(f))
    assert len(camera)==len(truth)==5
    assert len(imu)==81
    assert all(int(r["arrival_timestamp_imu_us"]) - int(r["timestamp_us"])==75000
               for r in camera)
    assert all(int(r["range_timestamp_us"])==int(r["timestamp_us"])
               for r in camera)
    assert float(imu[0]["accel_z_m_s2"]) == pytest.approx(-9.80665,abs=.05)
    assert float(truth[0]["vn_m_s"]) > 1.
    assert (folder/"frame_000.png").exists()
    assert (folder/"synthetic_orthophoto.png").exists()


def test_actual_vo_map_fixed_lag_eskf_pipeline_handles_100m_nominal(tmp_path):
    folder,report=_run(tmp_path,frames=5)
    assert report["kind"]=="IDEAL_PINHOLE_SYNTHETIC_SENSOR_BENCH"
    assert not report["flight_accuracy_validated"]
    assert not report["imu_failure_latched"]
    assert not report["applied_map_correction"]
    assert report["map_geometry_accepted"]>=3,report
    assert report["visual_velocity_accepted"]>=2,report
    assert report["visual_updates_fused"]>=2,report
    assert report["map_position_rmse_m"]<0.7,report
    assert report["visual_speed_rmse_m_s"]<0.6,report
    with (tmp_path/"output"/"sensor_bench_results.csv").open(newline="") as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==5
    assert all(x["applied_map_correction"]=="0" for x in rows)


def test_real_camera_rotations_are_generated_and_nadir_derotation_is_exercised(tmp_path):
    folder,result=_run(tmp_path,scenario="yaw_rotation",frames=5)
    assert result["camera_available"]==5
    with (folder/"camera.csv").open(newline="") as f:
        frames=list(csv.DictReader(f))
    assert float(frames[-1]["yaw_rad"])>0.02
    assert result["map_geometry_accepted"]>=2,result
    assert result["visual_velocity_accepted"]>=2,result


def test_outage_restarts_visual_segment_and_does_not_invent_missing_frames(tmp_path):
    folder,result=_run(tmp_path,scenario="camera_blackout",frames=6)
    assert result["camera_available"]==4
    with (tmp_path/"output"/"sensor_bench_results.csv").open(newline="") as f:
        rows=list(csv.DictReader(f))
    assert rows[2]["visual_status"]=="SYNTHETIC_CAMERA_UNAVAILABLE"
    assert rows[3]["visual_status"]=="SYNTHETIC_CAMERA_UNAVAILABLE"
    assert rows[4]["visual_status"]=="INITIALIZED"
    assert rows[4]["visual_segment_id"]!=rows[1]["visual_segment_id"]
    assert result["visual_velocity_accepted"]<=2


def test_duplicated_map_texture_at_wrong_coordinates_is_rejected(tmp_path):
    folder,result=_run(tmp_path,scenario="duplicate_map",frames=4)
    assert result["map_geometry_accepted"]==0
    assert result["map_rejected_ambiguity"]==4
    assert result["visual_velocity_accepted"]>=1
    assert not result["applied_map_correction"]


def test_imu_gap_latches_navigation_failure_even_if_camera_remains_operational(tmp_path):
    folder,result=_run(tmp_path,scenario="imu_gap",frames=5)
    assert result["imu_failure_latched"],result
    with (tmp_path/"output"/"sensor_bench_results.csv").open(newline="") as f:
        rows=list(csv.DictReader(f))
    assert rows[-1]["eskf_status"]=="IMU_FAILURE_LATCHED"
    assert result["map_geometry_accepted"]>=2


def test_range_skew_is_rejected_before_any_camera_processing(tmp_path):
    folder=generate_dataset(tmp_path/"input",BenchConfig(frames=4))
    path=folder/"camera.csv"
    with path.open(newline="") as f:
        r=csv.DictReader(f)
        fields=r.fieldnames
        rows=list(r)
    rows[2]["range_timestamp_us"]="1"
    with path.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError,match="не синхронизированы"):
        run_sensor_bench(folder,tmp_path/"output")


def test_dataset_seed_reproduces_exact_images_and_imu_records(tmp_path):
    one=generate_dataset(tmp_path/"one",BenchConfig(frames=4,seed=56))
    two=generate_dataset(tmp_path/"two",BenchConfig(frames=4,seed=56))
    assert (one/"imu.csv").read_bytes()==(two/"imu.csv").read_bytes()
    assert (one/"truth.csv").read_bytes()==(two/"truth.csv").read_bytes()
    assert np.array_equal(
        cv2.imread(str(one/"frame_001.png"),0),
        cv2.imread(str(two/"frame_001.png"),0)
    )


def test_invalid_motion_design_fails_before_render(tmp_path):
    with pytest.raises(ValueError):
        generate_dataset(tmp_path,BenchConfig(frames=2))
    with pytest.raises(ValueError):
        generate_dataset(tmp_path,BenchConfig(imu_dt_s=.013))
