"""PX4 SITL replay fixture is derived from VO + IMU only, never truth."""
import csv
import math
import json
import pytest

from src.synthetic_sensor_bench import BenchConfig, generate_dataset, run_sensor_bench
from src.export_px4_sitl_replay import export_replay, FIELDS


def build(tmp_path, scenario="nominal", frames=6):
    src = generate_dataset(
        tmp_path / "sensors",
        BenchConfig(scenario=scenario, frames=frames, seed=72118)
    )
    result = run_sensor_bench(src, tmp_path / "evaluation")
    out = tmp_path / "replay.csv"
    report = export_replay(
        tmp_path / "evaluation" / "sensor_bench_results.csv",
        src / "camera.csv", src / "imu.csv", out,
    )
    with out.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return report, rows, result


def test_high_altitude_fixture_preserves_100m_and_explicitly_marks_unsafe_fusion(tmp_path):
    report, rows, source = build(tmp_path)
    assert report["rows"] == 6
    assert report["flow_valid_rows"] == 5
    assert report["flow_invalid_rows"] == 1
    assert not report["publish_to_uorb"]
    assert not report["publish_to_ekf2"]
    assert tuple(rows[0]) == FIELDS
    assert all(float(x["height_m"]) == pytest.approx(100.) for x in rows)
    assert rows[0]["flow_valid"] == "0"
    assert all(x["flow_valid"] == "1" for x in rows[1:])
    assert all(
        int(x["arrival_us"])-int(x["timestamp_us"]) == 75000
        for x in rows
    )
    assert all(float(x["integration_dt_s"]) == pytest.approx(.2)
               for x in rows[1:])
    assert all(float(x["accel_z_m_s2"]) == pytest.approx(-9.80665, abs=.04)
               for x in rows[1:])


def test_flow_integrals_encode_vo_body_velocity_with_correct_axis_sign(tmp_path):
    report, rows, _ = build(tmp_path)
    for r in rows[1:]:
        h = float(r["height_m"])
        dt = float(r["integration_dt_s"])
        velocity_body_x = h * float(r["flow_y_rad"]) / dt
        velocity_body_y = -h * float(r["flow_x_rad"]) / dt
        # Existing C++ sensor convention: v_body_x = h*flow_y/dt,
        # v_body_y = -h*flow_x/dt, with zero X/Y gyro integrals.
        assert velocity_body_x > 0.
        assert math.isfinite(velocity_body_y)


def test_missing_imu_at_accepted_exposure_must_fail_closed(tmp_path):
    folder=generate_dataset(
        tmp_path/"sensors",
        BenchConfig(scenario="nominal", frames=4)
    )
    run_sensor_bench(folder,tmp_path/"evaluation")
    imu=folder/"imu.csv"
    with imu.open(newline="") as f:
        rdr=csv.DictReader(f)
        fields=rdr.fieldnames
        data=list(rdr)
    data=[r for r in data if int(r["timestamp_us"])!=1_200_000]
    with imu.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        w.writerows(data)
    with pytest.raises(ValueError,match="без ИИМ"):
        export_replay(
            tmp_path/"evaluation"/"sensor_bench_results.csv",
            folder/"camera.csv", imu, tmp_path/"replay.csv"
        )


def test_blackout_explicitly_reports_invalid_flow_without_forging_gyro(tmp_path):
    report,rows,result=build(tmp_path,"camera_blackout",6)
    assert report["flow_invalid_rows"]>=3
    assert rows[2]["flow_valid"]=="0"
    assert rows[3]["flow_valid"]=="0"
    assert rows[2]["quality"]=="0"
    assert rows[2]["flow_x_rad"]=="0.0"
    assert not result["imu_failure_latched"]


def test_replay_csv_excludes_map_and_reference_fields(tmp_path):
    _,rows,_=build(tmp_path,"nominal",4)
    assert not ({"north_m","east_m","map_n_m","map_e_m",
                "true_n_m","true_e_m"} & set(rows[0]))
