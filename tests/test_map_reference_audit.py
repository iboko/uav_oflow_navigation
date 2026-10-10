"""Independent map-fix statistics must count accepted but WRONG positions."""
import csv
import json

import pytest

from src.run_map_reference_audit import evaluate_flight, run_reference_audit


def c(t, n, status="GEOMETRICALLY_ACCEPTED_UNVALIDATED"):
    return {
        "timestamp_us": str(t), "map_status": status,
        "candidate_n_m": str(n), "candidate_e_m": "0",
    }


def truth(t, n=0., source="independent-survey"):
    return {
        "timestamp_us": str(t), "true_n_m": str(n),
        "true_e_m": "0", "source": source,
    }


def test_false_map_fix_is_counted_among_accepted_not_hidden_by_ransac():
    observations = [
        c(1_000_000, .3),
        c(1_100_000, 35.),
        c(1_200_000, 0., "AMBIGUOUS_MAP_MATCH"),
    ]
    reference = [truth(1_000_000), truth(1_100_000), truth(1_200_000)]
    result, rows = evaluate_flight(
        observations, reference, false_fix_error_threshold_m=3.
    )
    assert result["frames_evaluated"] == 3
    assert result["geometrically_accepted"] == 2
    assert result["false_fixes"] == 1
    assert result["false_fix_rate_among_accepted"] == pytest.approx(.5)
    assert result["false_fix_fraction_of_all_frames"] == pytest.approx(1/3)
    assert rows[1]["false_fix"] == 1
    assert rows[2]["error_ne_m"] is None


def test_missing_reference_must_reject_without_silent_interpolation():
    with pytest.raises(ValueError, match="не у всех"):
        evaluate_flight(
            [c(1_000_000, 0.), c(1_100_000, 0.)],
            [truth(1_000_000)],
            false_fix_error_threshold_m=2.
        )


def test_self_generated_truth_is_rejected():
    with pytest.raises(ValueError, match="независимая"):
        evaluate_flight([c(1_000_000, 0.)], [truth(1_000_000, source="estimator")],
                        false_fix_error_threshold_m=3.)


def test_no_accepted_fixes_produces_no_fake_zero_rmse():
    info, _ = evaluate_flight(
        [c(1_000_000, 0., "INSUFFICIENT_TEXTURE")],
        [truth(1_000_000)], false_fix_error_threshold_m=3.
    )
    assert info["geometrically_accepted"] == 0
    assert info["false_fix_rate_among_accepted"] is None
    assert info["rmse_ne_m_on_accepted"] is None


def test_reference_audit_multiple_flights_exports_independent_summary(tmp_path):
    manifest = tmp_path/"flights.csv"
    with manifest.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["flight_id", "candidate_csv", "truth_csv"])
        for i in range(3):
            name = f"F{i+1}"
            candidates = tmp_path/f"{name}_candidates.csv"
            reference = tmp_path/f"{name}_truth.csv"
            with candidates.open("w", encoding="utf-8", newline="") as cf:
                cw = csv.DictWriter(cf,fieldnames=[
                    "timestamp_us","map_status","candidate_n_m","candidate_e_m"
                ])
                cw.writeheader()
                for j in range(4):
                    cw.writerow(c(1_000_000+j*100_000,
                                 5. if i==2 and j==0 else .1))
            with reference.open("w", encoding="utf-8", newline="") as tf:
                tw = csv.DictWriter(tf,fieldnames=[
                    "timestamp_us","true_n_m","true_e_m","source"
                ])
                tw.writeheader()
                for j in range(4):
                    tw.writerow(truth(1_000_000+j*100_000))
            w.writerow([name,candidates.name,reference.name])
    result=run_reference_audit(
        manifest, tmp_path/"out", false_fix_error_threshold_m=2.
    )
    assert result["independent_flights"]==3
    assert result["flights"]["F3"]["false_fixes"]==1
    assert result["flights"]["F1"]["false_fixes"]==0
    assert result["mean_false_fix_rate_over_flights"]==pytest.approx(1/12)
    assert not result["validated_for_flight"]
    assert (tmp_path/"out"/"map_reference_errors.csv").exists()
    assert json.loads((tmp_path/"out"/"map_reference_summary.json").read_text())[
        "independent_flights"
    ]==3
