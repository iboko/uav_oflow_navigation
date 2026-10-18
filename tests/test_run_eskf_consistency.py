"""End-to-end independent-reference CSV/JSONL validation."""
import csv
import json
from dataclasses import replace

import numpy as np
import pytest

from src.run_eskf_consistency import run_consistency


def _record(t, n=0.):
    return {
        "timestamp_us": t,
        "status": "VISUAL_CORRECTED",
        "position_ned_m": [0., 0., 0.],
        "velocity_ned_m_s": [n, 0., 0.],
        "quaternion_body_to_ned": [1., 0., 0., 0.],
        "gyro_bias_rad_s": [0., 0., 0.],
        "accel_bias_m_s2": [0., 0., 0.],
        "covariance_15x15": np.diag(
            [2.]*3 + [0.04]*3 + [0.01]*3 +
            [0.0004]*3 + [0.0025]*3
        ).tolist(),
        "visual_status": "VISUAL_CORRECTED",
    }


def make_flight(tmp_path, name="F01", *, include_biases=True):
    states = tmp_path / f"{name}_states.jsonl"
    truth = tmp_path / f"{name}_truth.csv"
    time_values = [1_000_000 + i * 10_000 for i in range(35)]
    with states.open("w", encoding="utf-8") as f:
        for t in time_values:
            f.write(json.dumps(_record(t, n=0.2))+"\n")
    columns = [
        "timestamp_us","vn_m_s","ve_m_s","vd_m_s","source",
        "north_m","east_m","down_m","qw","qx","qy","qz",
    ]
    if include_biases:
        columns += [f"bg_{a}_rad_s" for a in "xyz"]
        columns += [f"ba_{a}_m_s2" for a in "xyz"]
    with truth.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for t in time_values:
            row = {
                "timestamp_us": t, "vn_m_s": 0., "ve_m_s": 0.,
                "vd_m_s": 0., "source": "independent-motion-capture",
                "north_m": 0., "east_m": 0., "down_m": 0.,
                "qw": 1., "qx": 0., "qy": 0., "qz": 0.,
            }
            if include_biases:
                row.update({c: 0. for c in columns if c.startswith(("bg_", "ba_"))})
            writer.writerow(row)
    return states, truth


def _manifest(tmp_path, specs, *, origin="1", heading="1"):
    path = tmp_path / "flight_manifest.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "flight_id", "states_jsonl", "truth_csv",
            "nis_csv", "linearization_csv",
            "origin_alignment_verified", "heading_alignment_verified",
        ])
        writer.writeheader()
        for name, states, truth, nis, lin in specs:
            writer.writerow({
                "flight_id": name, "states_jsonl": states.name,
                "truth_csv": truth.name, "nis_csv": nis or "",
                "linearization_csv": lin or "",
                "origin_alignment_verified": origin,
                "heading_alignment_verified": heading,
            })
    return path


def test_three_independent_flights_report_nees_and_full_conditional_nees(tmp_path):
    specs=[]
    for idx in range(3):
        name=f"F0{idx+1}"
        states, truth=make_flight(tmp_path,name)
        specs.append((name,states,truth,"",""))
    report=run_consistency(_manifest(tmp_path,specs),tmp_path/"result")
    assert report["independent_flights"]==3
    assert report["sufficient_for_preliminary_screening"]
    assert report["total_matched_samples"]==105
    assert not report["validated"]
    assert report["pooled_velocity_rmse_m_s"]==pytest.approx(0.2)
    assert report["flight_bootstrap_95pct_interval_for_mean"] is not None
    for entry in report["data_reconciliation"].values():
        assert entry["full_nees_15d_count"]==35
        assert entry["mean_full_nees_15d_conditional"]==pytest.approx(1.)
        assert entry["unmatched_estimated_epochs"]==0
    with (tmp_path/"result"/"eskf_reference_errors.csv").open(newline="") as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==105
    assert float(rows[0]["nees_vne_2d"])==pytest.approx(1.)


def test_gauge_not_verified_disables_full_nees_even_with_complete_truth(tmp_path):
    states, truth=make_flight(tmp_path)
    report=run_consistency(
        _manifest(tmp_path,[("F01",states,truth,"","")],origin="0"),
        tmp_path/"out"
    )
    assert report["data_reconciliation"]["F01"]["full_nees_15d_count"]==0
    assert not report["sufficient_for_preliminary_screening"]


def test_nis_count_includes_gated_outlier_and_observability_flags_gauge(tmp_path):
    states,truth=make_flight(tmp_path)
    nis=tmp_path/"nis.csv"
    with nis.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=["status","innovation_nis"])
        writer.writeheader()
        writer.writerows([
            {"status":"VISUAL_CORRECTED","innovation_nis":1.},
            {"status":"VISUAL_OUTLIER_REJECTED","innovation_nis":20.},
            {"status":"INVALID_VISUAL_MEASUREMENT","innovation_nis":"nan"}
        ])
    linear=tmp_path/"linear.csv"
    with linear.open("w",newline="",encoding="utf-8") as f:
        cols=["dt_s","qw","qx","qy","qz",
              "wx_rad_s","wy_rad_s","wz_rad_s",
              "fx_m_s2","fy_m_s2","fz_m_s2"]
        writer=csv.DictWriter(f,fieldnames=cols)
        writer.writeheader()
        for _ in range(20):
            writer.writerow({"dt_s":0.01,"qw":1,"qx":0,"qy":0,"qz":0,
                             "wx_rad_s":0,"wy_rad_s":0,"wz_rad_s":0,
                             "fx_m_s2":0,"fy_m_s2":0,"fz_m_s2":-9.80665})
    manifest=_manifest(tmp_path,[("F01",states,truth,nis.name,linear.name)])
    report=run_consistency(manifest,tmp_path/"out")
    info=report["data_reconciliation"]["F01"]
    assert info["pre_gate_nis"]["pre_gate_innovations"]==2
    assert info["pre_gate_nis"]["mean_nis_2d"]==pytest.approx(10.5)
    assert info["pre_gate_nis"]["fraction_above_chi2_2_99"]==pytest.approx(0.5)
    assert info["local_observability"]["rank"]<15
    assert info["local_observability"]["translation_gauge_unobservable"]


def test_no_exact_time_matches_is_error_not_hidden_interpolation(tmp_path):
    states,truth=make_flight(tmp_path)
    with truth.open(newline="",encoding="utf-8") as f:
        reader=csv.DictReader(f)
        columns=reader.fieldnames
        rows=list(reader)
    for r in rows:
        r["timestamp_us"]=str(int(r["timestamp_us"])+1)
    with truth.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError,match="одной пары"):
        run_consistency(_manifest(tmp_path,[("F01",states,truth,"","")]),
                        tmp_path/"out")


def test_user_declared_self_generated_ground_truth_is_rejected(tmp_path):
    states,truth=make_flight(tmp_path)
    with truth.open(newline="",encoding="utf-8") as f:
        reader=csv.DictReader(f)
        columns=reader.fieldnames
        rows=list(reader)
    for r in rows:
        r["source"]="estimator"
    with truth.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError,match="независимый"):
        run_consistency(_manifest(tmp_path,[("F01",states,truth,"","")]),
                        tmp_path/"out")
