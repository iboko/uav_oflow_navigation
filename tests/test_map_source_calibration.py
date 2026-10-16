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
                              place_weighted=True, quality_model=False)
    mf.combine(args)
    cal = json.loads((tmp_path / "cal" / "map_calibration.json").read_text())
    assert cal["covariance_model"] == "place-weighted isotropic"
    assert cal["fit_flights"] == ["f1", "f2", "f3", "f4"] and cal["holdout_flights"] == ["f5", "f6"]
    assert cal["within_flight_covariance_ne_m2"] is None
    assert cal["covariance_ne_m2"] == cal["place_model"]["covariance_ne_m2"]
    assert len(cal["holdout_check"]["per_flight_mean_nees"]) == 2
    assert "cross_validation_4_fit" in cal["place_model"]
