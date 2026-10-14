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
