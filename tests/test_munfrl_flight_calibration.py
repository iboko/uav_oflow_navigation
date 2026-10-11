"""Калибровка по нескольким полетам: известная модель шума восстанавливается."""
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
_SPEC = importlib.util.spec_from_file_location("munfrl_flight_calibration",
                                               TOOLS / "munfrl_flight_calibration.py")
cal = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cal)

TRUE = {"sigma_min_m_s": 0.10, "sigma_rate_rad_s": 0.02, "sigma_h_rel": 0.05}


def _rows(seed: int, heights: tuple[float, float], heading_deg: float, scale: float,
          noise_follows_scaled_height: bool = False) -> list[dict]:
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(400):
        h = rng.uniform(*heights)
        speed = rng.uniform(0.0, 6.0)
        course = rng.uniform(-math.pi, math.pi)
        ref = speed * np.array([math.cos(course), math.sin(course)])
        sigma = math.sqrt(cal.sigma2(TRUE, h, speed))
        measured = ref + rng.normal(0.0, sigma, 2)
        # The attitude source is rotated and the speed is scaled. The noise
        # follows the reported height ("assumed") or the true height when
        # the scale is a terrain height error ("effective").
        vo = cal.rot(heading_deg) @ measured * scale
        reported = h * scale if noise_follows_scaled_height else h
        rows.append({"vo": vo, "ref": ref, "h": reported, "dt": 0.1})
    return rows


def _flights(model: str = "assumed"):
    specs = [("a", 1, (10., 25.), -60., 1.08), ("b", 2, (25., 40.), 15., 0.95),
             ("c", 3, (40., 60.), 100., 1.02)]
    return [cal.align_flight(n, _rows(s, hs, hd, sc, model == "effective"), height_model=model)
            for n, s, hs, hd, sc in specs]


def test_alignment_recovers_heading_and_scale_per_flight():
    flights = _flights()
    for flight, (heading, scale) in zip(flights, ((-60., 1.08), (15., .95), (100., 1.02))):
        assert flight["heading_offset_deg"] == pytest.approx(heading, abs=1.5)
        assert flight["speed_scale"] == pytest.approx(scale, rel=0.03)


@pytest.mark.parametrize("model", cal.HEIGHT_MODELS)
def test_noise_model_is_recovered_and_holdout_nees_is_consistent(model):
    flights = _flights(model)
    model = cal.fit_noise_model(flights)
    assert model["identifiable"]
    # The dominant height term is resolved and recovered.
    assert "sigma_rate_rad_s" in model["resolved_terms"]
    assert model["sigma_rate_rad_s"] == pytest.approx(TRUE["sigma_rate_rad_s"], rel=0.15)
    # Small terms: every coefficient is within 3 SE of the truth.
    truth = [TRUE["sigma_min_m_s"] ** 2, TRUE["sigma_rate_rad_s"] ** 2, TRUE["sigma_h_rel"] ** 2]
    for c, se, t in zip(model["variance_coefficients"], model["variance_coefficient_se"], truth):
        assert abs(c - t) <= 3.0 * se + 1e-6
    folds = cal.leave_one_flight_out(flights)
    assert [f["holdout_flight"] for f in folds] == ["a", "b", "c"]
    for fold in folds:
        assert 1.6 < fold["holdout"]["mean_nees_2d"] < 2.6
        # The design constants of the profile underestimate this noise.
        assert fold["profile_constants_on_holdout"]["mean_nees_2d"] > 10.


def test_too_few_fast_intervals_is_refused():
    rows = _rows(4, (20., 30.), 0., 1.)
    slow = [dict(r, ref=r["ref"] * 0.1, vo=r["vo"] * 0.1) for r in rows]
    with pytest.raises(ValueError, match="быстрее 2 м/с"):
        cal.align_flight("slow", slow)


def test_unknown_height_model_is_refused():
    with pytest.raises(ValueError, match="модель высоты"):
        cal.align_flight("a", _rows(5, (20., 30.), 0., 1.), height_model="terrain")
