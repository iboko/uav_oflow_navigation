"""Reproducible offline CI candidate: never mutates ESKF or trusts YAML alone."""
import csv
import json

import pytest

from src.run_map_ci_offline import run_map_ci_offline


def make_inputs(tmp_path, *, declared=True):
    cal=tmp_path/"map_error_calibration.json"
    cal.write_text(json.dumps({
        "bias_ne_m":[.2,-.1],
        "covariance_ne_m2":[[.5,.02],[.02,.4]],
        "covariance_model_status":"EMPIRICAL_HOLDOUT_UNCERTIFIED",
        "flightworthiness_approved":False
    }))
    ver=tmp_path/"checks.yaml"
    ver.write_text(
        f"prior_covariance_independently_validated: {str(declared).lower()}\n"
        f"map_covariance_independently_validated: {str(declared).lower()}\n"
        f"reference_frames_and_timestamps_aligned: {str(declared).lower()}\n"
        f"independent_map_integrity_review: {str(declared).lower()}\n"
        "max_position_disagreement_m: 20\n",
        encoding="utf-8"
    )
    mapfile=tmp_path/"map.csv"
    with mapfile.open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f)
        w.writerow(["timestamp_us","map_status","candidate_n_m","candidate_e_m",
                    "temporal_integrity_status"])
        for i in range(3):
            w.writerow([
                1_000_000+i*100_000,"GEOMETRICALLY_ACCEPTED_UNVALIDATED",
                10.2+i*.1,-2.1,
                "MAP_TEMPORALLY_CONSISTENT_UNVALIDATED"
            ])
    prior=tmp_path/"prior.csv"
    with prior.open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f)
        w.writerow(["timestamp_us","prior_n_m","prior_e_m",
                   "prior_cov_nn_m2","prior_cov_ne_m2","prior_cov_ee_m2"])
        for i in range(3):
            w.writerow([1_000_000+i*100_000,10+i*.1,-2., 1.,0.,1.])
    return prior,mapfile,cal,ver


def test_ci_runner_corrects_train_bias_and_never_mutates_eskf(tmp_path):
    prior,mapfile,cal,ver=make_inputs(tmp_path)
    result=run_map_ci_offline(prior,mapfile,cal,ver,tmp_path/"out")
    assert result["frames"]==3
    assert result["statuses"]=={"OFFLINE_CI_CANDIDATE_NOT_APPLIED":3}
    assert result["applied_to_eskf"]==0
    with (tmp_path/"out"/"map_ci_candidates.csv").open(newline="") as f:
        rows=list(csv.DictReader(f))
    assert all(row["applied_to_eskf"]=="0" for row in rows)
    assert float(rows[0]["candidate_n_m"])==pytest.approx(10.,abs=1e-10)
    assert float(rows[0]["candidate_e_m"])==pytest.approx(-2.,abs=1e-10)


def test_no_independent_covariance_flags_prevents_ci(tmp_path):
    prior,mapfile,cal,ver=make_inputs(tmp_path,declared=False)
    result=run_map_ci_offline(prior,mapfile,cal,ver,tmp_path/"out")
    assert result["statuses"]=={"MAP_INTEGRITY_NOT_ESTABLISHED":3}
    assert result["applied_to_eskf"]==0


def test_mismatched_timestamps_fail_closed_without_interpolation(tmp_path):
    prior,mapfile,cal,ver=make_inputs(tmp_path)
    with mapfile.open(newline="",encoding="utf-8") as f:
        rows=list(csv.reader(f))
    rows[1][0]="1000001"
    with mapfile.open("w",newline="",encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    with pytest.raises(ValueError,match="Несовпадающие временные метки"):
        run_map_ci_offline(prior,mapfile,cal,ver,tmp_path/"out")


def test_calibration_json_cannot_claim_flight_approval(tmp_path):
    prior,mapfile,cal,ver=make_inputs(tmp_path)
    data=json.loads(cal.read_text())
    data["flightworthiness_approved"]=True
    cal.write_text(json.dumps(data))
    with pytest.raises(ValueError,match="не должна разрешать полет"):
        run_map_ci_offline(prior,mapfile,cal,ver,tmp_path/"out")
