"""Train/holdout calibration requires independent sorties, not frames."""
import csv
import json
import numpy as np
import pytest

from src.map_error_calibration import MapErrorFlight, calibrate_map_error
from src.run_map_error_calibration import run_map_calibration


def flight(name, count=50, bias=(.4,-.2), phase=0., error=False):
    times=np.arange(count,dtype=float)
    residual=np.column_stack((
        np.sin(times*.37+phase)*.4 + bias[0],
        np.cos(times*.31+phase)*.6 + bias[1]
    ))
    if error:
        residual[0,0] += 15.
    return MapErrorFlight(name,tuple(map(tuple,residual)), count, int(error))


def test_separate_flights_yield_centered_covariance_and_holdout_coverage():
    tr=[flight(f"T{i}",phase=i*.1) for i in range(3)]
    va=[flight(f"H{i}",phase=.25+i*.13) for i in range(2)]
    result=calibrate_map_error(tr,va)
    assert np.allclose(result.bias_ne_m,[.4,-.2],atol=.06)
    cov=np.asarray(result.covariance_ne_m2)
    assert np.linalg.eigvalsh(cov).min()>0
    assert result.fit_flights==("T0","T1","T2")
    assert result.holdout_flights==("H0","H1")
    assert result.holdout_nees_coverage_95pct>.7
    assert result.holdout_mean_nees_2d>0
    assert not result.independently_validated_for_flight


def test_held_out_outliers_are_reported_not_fit_away():
    tr=[flight(f"T{i}",phase=i*.15) for i in range(3)]
    va=[flight("H0",error=True), flight("H1")]
    result=calibrate_map_error(tr,va)
    assert result.holdout_false_fix_rate_among_accepted==pytest.approx(1/100)
    assert result.holdout_flight_details["H0"]["false_fixes"]==1
    assert result.holdout_flight_details["H0"]["mean_nees_2d"] > (
        result.holdout_flight_details["H1"]["mean_nees_2d"]
    )


def test_overlap_between_train_and_holdout_is_forbidden():
    training=[flight("A"),flight("B"),flight("C")]
    with pytest.raises(ValueError,match="пересекаются"):
        calibrate_map_error(training,[flight("C"),flight("D")])


def test_singular_calibration_does_not_invent_noise_floor():
    zero=tuple((.4,-.2) for _ in range(50))
    training=[MapErrorFlight(f"T{i}",zero,50,0) for i in range(3)]
    with pytest.raises(ValueError,match="положительно определена"):
        calibrate_map_error(training,[flight("H0"),flight("H1")])


def test_bad_total_frame_count_and_nonfinite_error_fail():
    training=[flight("T0"),flight("T1"),flight("T2")]
    bad=MapErrorFlight("H0",tuple([(float("nan"),0.)]*50),50,0)
    with pytest.raises(ValueError,match="конечными"):
        calibrate_map_error(training,[bad,flight("H1")])
    bad=MapErrorFlight("H0",training[0].errors_ne_m,5,0)
    with pytest.raises(ValueError,match="полнота"):
        calibrate_map_error(training,[bad,flight("H1")])


def test_csv_runner_exports_unvalidated_model_without_using_holdout_for_fit(tmp_path):
    manifest=tmp_path/"map_training.csv"
    with manifest.open("w",newline="",encoding="utf-8") as f:
        writer=csv.writer(f)
        writer.writerow(["flight_id","partition","candidate_csv","truth_csv"])
        for i in range(5):
            fid=f"F{i}"
            trained=i<3
            flight_data=flight(fid,phase=i*.09)
            candidates=tmp_path/f"{fid}_candidates.csv"
            truth=tmp_path/f"{fid}_truth.csv"
            with candidates.open("w",newline="",encoding="utf-8") as c:
                w=csv.writer(c)
                w.writerow(["timestamp_us","map_status","candidate_n_m","candidate_e_m"])
                for j,(n,e) in enumerate(flight_data.errors_ne_m):
                    w.writerow([1_000_000+j*10000,
                         "GEOMETRICALLY_ACCEPTED_UNVALIDATED",n,e])
            with truth.open("w",newline="",encoding="utf-8") as t:
                w=csv.writer(t)
                w.writerow(["timestamp_us","true_n_m","true_e_m","source"])
                for j in range(50):
                    w.writerow([1_000_000+j*10000,0,0,"external-motion-capture"])
            writer.writerow([fid,"train" if trained else "holdout",
                             candidates.name,truth.name])
    result=run_map_calibration(
        manifest,tmp_path/"out",false_fix_error_threshold_m=3.
    )
    assert len(result["fit_flights"])==3
    assert len(result["holdout_flights"])==2
    assert not result["flightworthiness_approved"]
    assert result["covariance_model_status"]=="EMPIRICAL_HOLDOUT_UNCERTIFIED"
    stored=json.loads((tmp_path/"out"/"map_error_calibration.json").read_text())
    assert stored["holdout_flights"]==["F3","F4"]
    assert np.linalg.eigvalsh(stored["covariance_ne_m2"]).min()>0
