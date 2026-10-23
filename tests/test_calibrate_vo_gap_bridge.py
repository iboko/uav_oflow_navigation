"""VO-gap bridge calibration: fit of the gap-length error model."""
import numpy as np
import pytest

from tools.calibrate_vo_gap_bridge import _nnls, evaluate, fit, sigma


def _flight(rng, params, n):
    gap = rng.uniform(0.3, 4.0, n)
    s = sigma(params, gap)
    return np.column_stack([gap, rng.normal(0, s), rng.normal(0, s)])


def test_fit_recovers_the_error_model_and_is_consistent():
    rng = np.random.default_rng(1)
    truth = (0.2, 0.15, 0.0)
    flights = [_flight(rng, truth, 400) for _ in range(3)]
    p = fit(flights)
    assert sigma(p, 3.0) == pytest.approx(sigma(truth, 3.0), rel=0.1)
    ev = evaluate(p, _flight(rng, truth, 400))
    assert ev["mean_nees_2d"] == pytest.approx(2.0, rel=0.15)
    assert ev["coverage_95"] == pytest.approx(0.95, abs=0.03)


def test_nnls_keeps_coefficients_non_negative():
    a = np.column_stack([np.ones(5), np.arange(5.0)])
    b = 3.0 - np.arange(5.0)          # unconstrained slope would be negative
    x = _nnls(a, b)
    assert np.all(x >= 0) and x[1] == 0.0 and x[0] == pytest.approx(1.0)
