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


def test_training_false_match_invalidates_single_gaussian_model():
    bad = flight("T0", error=True)
    training = [bad, flight("T1"), flight("T2")]
    with pytest.raises(ValueError, match="ложные принятые"):
        calibrate_map_error(training, [flight("H0"), flight("H1")])


def test_holdout_flights_weighted_equally_independent_of_camera_frame_rate():
    training = [flight("T0"), flight("T1"), flight("T2")]
    normal = flight("H0", count=50)
    different = flight("H1", count=100, bias=(1.0, 1.0))
    result = calibrate_map_error(training, [normal, different])
    expected = (
        result.holdout_flight_details["H0"]["mean_nees_2d"] +
        result.holdout_flight_details["H1"]["mean_nees_2d"]
    ) / 2.0
    assert result.holdout_mean_nees_2d == pytest.approx(expected)
    expected_cover = (
        result.holdout_flight_details["H0"]["fraction_within_chi2_2_95"] +
        result.holdout_flight_details["H1"]["fraction_within_chi2_2_95"]
    ) / 2.0
    assert result.holdout_nees_coverage_95pct == pytest.approx(expected_cover)


def test_covariance_splits_into_within_and_between_flight_parts():
    # Per-flight offsets (an attitude tilt error at the calibration height)
    # appear in the between-flight part only.
    tr = [flight("T0", bias=(1.0, 0.0)), flight("T1", bias=(-1.0, 0.0), phase=.1),
          flight("T2", bias=(0.0, 0.0), phase=.2)]
    va = [flight(f"H{i}", phase=.25 + i * .13) for i in range(2)]
    r = calibrate_map_error(tr, va)
    total, within, between = (np.asarray(x) for x in (r.covariance_ne_m2, r.within_flight_covariance_ne_m2,
                                                      r.between_flight_covariance_ne_m2))
    assert np.allclose(total, within + between)
    assert np.isclose(between[0, 0], 2 / 3, atol=0.01)
    assert within[0, 0] < 0.2


def test_fix_covariance_scales_tilt_part_with_height():
    from src.map_error_calibration import covariance_at_height
    cal = {"covariance_ne_m2": [[0.5, 0], [0, 0.5]], "within_flight_covariance_ne_m2": [[0.1, 0], [0, 0.1]],
           "between_flight_covariance_ne_m2": [[0.4, 0], [0, 0.4]], "calibration_height_m": 40.0}
    assert np.allclose(covariance_at_height(cal, 40.0), [[0.5, 0], [0, 0.5]])
    assert np.allclose(covariance_at_height(cal, 80.0), [[1.7, 0], [0, 1.7]])
    # Old calibration files without the decomposition: unchanged.
    assert np.allclose(covariance_at_height({"covariance_ne_m2": [[0.5, 0], [0, 0.5]]}, 80.0), 0.5 * np.eye(2))
    with pytest.raises(ValueError):
        covariance_at_height(cal, float("nan"))


def test_quality_model_recovers_scale_of_subpatch_covariance():
    from src.map_error_calibration import fit_quality_model, quality_covariance
    rng = np.random.default_rng(3)

    def flight(n=200):
        s = rng.uniform(0.05, 1.5, n)                    # sub-patch scatter varies per fix
        c = np.stack([np.diag([v, 0.3 * v]) for v in s])  # anisotropic
        r = 0.04 * np.eye(2)[None] + 0.5 * c              # truth: floor 0.04, k2 0.5
        e = np.stack([rng.multivariate_normal([0.2, -0.1], ri) for ri in r])
        return e, c
    fit = [flight() for _ in range(3)]
    hold = [flight() for _ in range(2)]
    m = fit_quality_model([f[0] for f in fit], [f[1] for f in fit], [h[0] for h in hold], [h[1] for h in hold],
                          (0.2, -0.1))
    assert 0.3 < m["k2"] < 0.8 and 0.01 < m["floor_m2"] < 0.1, m
    assert all(1.5 < x < 2.6 for x in m["holdout_mean_nees"])
    assert np.allclose(quality_covariance(0.04, 0.5, [[1, 0], [0, 0.3]]), [[0.54, 0], [0, 0.19]])


def test_place_rule_separates_false_places_and_false_fixes():
    from src.map_error_calibration import place_outlier_rule
    rng = np.random.default_rng(1)
    pos = np.column_stack([np.repeat(np.arange(6) * 20.0 + 10, 30), np.zeros(180)])
    offs = np.array([[0.5, 0], [0.8, 0.2], [-0.3, 0.1], [12.0, -5.0], [0.2, -0.4], [1.0, 0.0]])
    e = np.repeat(offs, 30, axis=0) + rng.normal(0, 0.15, (180, 2))
    e[5] += (3.0, 0.0)                          # a single false fix in place 0
    r = place_outlier_rule(e, pos)
    assert r["places"] == 6 and r["false_places"] == 1
    assert r["false_place"][90:120].all() and not r["false_place"][:90].any()
    assert r["false_fix"][5] and r["false_fix"].sum() == 1
    assert r["inlier"].sum() == 180 - 30 - 1
