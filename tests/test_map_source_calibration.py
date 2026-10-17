"""Per-source (ground-only / full map) bias and covariance from flight files."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))


def test_source_calibration_separates_biases(tmp_path):
    import gz_map_companion_flight as mf
    rng = np.random.default_rng(0)
    dirs = []
    for i in range(5):
        d = tmp_path / f"f{i + 1}"
        d.mkdir()
        e_g = rng.normal(0.0, 0.2, (40, 2))
        e_f = rng.normal(0.0, 0.5, (40, 2)) + (0.75, 0.0)
        (d / "map_errors.json").write_text(json.dumps({
            "errors_ne_m": np.vstack([e_g, e_f]).tolist(),
            "map_source": ["primary"] * 40 + ["fallback"] * 40}))
        dirs.append(d)
    out = mf.source_calibration(dirs)
    assert abs(out["bias"]["fallback"][0] - 0.75) < 0.15 and abs(out["bias"]["primary"][0]) < 0.1
    assert out["check"]["primary"]["sigma_m"] < out["check"]["fallback"]["sigma_m"]
    assert all(1.4 < x < 2.8 for x in out["check"]["fallback"]["holdout_mean_nees"])
    assert mf.source_calibration(dirs[:0] + dirs) is not None


def test_combine_place_weighted_with_more_fit_flights(tmp_path):
    import argparse
    import gz_map_companion_flight as mf
    rng = np.random.default_rng(3)
    dirs = []
    for i in range(6):
        d = tmp_path / f"f{i + 1}"
        d.mkdir()
        e, p = [], []
        for k in range(5):
            n = int(rng.integers(10, 60))
            e.append(rng.normal(0.0, 0.5, 2) + rng.normal(0.0, 0.1, (n, 2)))
            p.append(np.array([20.0 * (5 * i + k) + 10.0, 10.0]) + rng.uniform(-5, 5, (n, 2)))
        e, p = np.concatenate(e), np.concatenate(p)
        (d / "map_errors.json").write_text(json.dumps({
            "name": d.name, "errors_ne_m": e.tolist(), "positions_ne_m": p.tolist(),
            "total_frames": len(e), "false_fixes": 0, "outlier_rule": "place",
            "map_source": ["primary"] * len(e)}))
        dirs.append(d)
    args = argparse.Namespace(flights=dirs, out=tmp_path / "cal", fit_flights=4, isotropic=True,
                              place_weighted=True, quality_model=False, exclude_test_zone=False, flight_quantile=None)
    mf.combine(args)
    cal = json.loads((tmp_path / "cal" / "map_calibration.json").read_text())
    assert cal["covariance_model"] == "place-weighted isotropic"
    assert cal["fit_flights"] == ["f1", "f2", "f3", "f4"] and cal["holdout_flights"] == ["f5", "f6"]
    assert cal["within_flight_covariance_ne_m2"] is None
    assert cal["covariance_ne_m2"] == cal["place_model"]["covariance_ne_m2"]
    assert len(cal["holdout_check"]["per_flight_mean_nees"]) == 2
    assert "cross_validation_4_fit" in cal["place_model"]


def test_combine_leaves_out_fixes_in_the_test_zone(tmp_path):
    import argparse
    import gz_map_companion_flight as mf
    rng = np.random.default_rng(4)
    dirs = []
    for i in range(5):
        d = tmp_path / f"f{i + 1}"
        d.mkdir()
        # Half of the places in the test zone (north < 10, east > -10) with a
        # large offset: they must not enter the calibration.
        e, p = [], []
        for k in range(6):
            in_zone = k % 2 == 0
            centre = np.array([-30.0, 20.0 + 25 * k]) if in_zone else np.array([50.0, -50.0 - 20 * k])
            n = 20
            e.append((np.array([3.0, 0.0]) if in_zone else rng.normal(0, 0.3, 2)) + rng.normal(0, 0.05, (n, 2)))
            p.append(centre + rng.uniform(-4, 4, (n, 2)))
        e, p = np.concatenate(e), np.concatenate(p)
        (d / "map_errors.json").write_text(json.dumps({
            "name": d.name, "errors_ne_m": e.tolist(), "positions_ne_m": p.tolist(),
            "total_frames": len(e), "false_fixes": 0, "outlier_rule": "place"}))
        dirs.append(d)
    args = argparse.Namespace(flights=dirs, out=tmp_path / "cal", fit_flights=3, isotropic=True,
                              place_weighted=False, quality_model=False, exclude_test_zone=True, flight_quantile=None)
    mf.combine(args)
    cal = json.loads((tmp_path / "cal" / "map_calibration.json").read_text())
    assert cal["excluded_test_zone_fixes"] == {"fixes_per_flight": [60] * 5, "dropped_flights": []}
    assert abs(cal["bias_ne_m"][0]) < 0.5           # the 3 m zone offset is not in the bias
    assert cal["place_model"]["places"] == 9
    assert mf.in_test_zone(-30.0, 20.0) and not mf.in_test_zone(40.0, -40.0)


def test_flight_level_scale_uses_fit_flights_only(tmp_path):
    import argparse
    import gz_map_companion_flight as mf
    rng = np.random.default_rng(5)
    dirs = []
    for i in range(7):
        d = tmp_path / f"f{i + 1}"
        d.mkdir()
        # Regional offsets: a whole flight shifted, as in the swiss series.
        region = rng.normal(0.0, 0.6, 2)
        e, p = [], []
        for k in range(5):
            n = 30
            e.append(region + rng.normal(0.0, 0.3, 2) + rng.normal(0.0, 0.1, (n, 2)))
            p.append(np.array([50.0 + 20 * k, -50.0 - 20 * i]) + rng.uniform(-4, 4, (n, 2)))
        e, p = np.concatenate(e), np.concatenate(p)
        (d / "map_errors.json").write_text(json.dumps({
            "name": d.name, "errors_ne_m": e.tolist(), "positions_ne_m": p.tolist(),
            "total_frames": len(e), "false_fixes": 0, "outlier_rule": "place"}))
        dirs.append(d)
    common = dict(flights=dirs, fit_flights=5, isotropic=True, place_weighted=False, quality_model=False,
                  exclude_test_zone=False)
    mf.combine(argparse.Namespace(out=tmp_path / "a", flight_quantile=None, **common))
    mf.combine(argparse.Namespace(out=tmp_path / "b", flight_quantile=0.95, **common))
    a = json.loads((tmp_path / "a" / "map_calibration.json").read_text())
    b = json.loads((tmp_path / "b" / "map_calibration.json").read_text())
    scale = b["covariance_ne_m2"][0][0] / a["covariance_ne_m2"][0][0]
    assert scale >= 1.0 and "flight-level scale" in b["covariance_model"]
    assert b["holdout_check"]["before_scale"]["per_flight_mean_nees"] == a["holdout_check"]["per_flight_mean_nees"]
