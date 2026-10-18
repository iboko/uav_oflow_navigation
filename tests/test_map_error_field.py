"""Kriged map-error field: variogram, prediction, flight variance, consistency."""
import numpy as np
import pytest

from src.map_error_field import (MapErrorField, cross_validated_field_nees, field_nees, fit_field,
                                 fit_variogram, place_offsets)


def _smooth_field(n, e):
    # Offset varying over ~60 m, like the swiss map (sd ~0.6 m).
    return np.stack([0.6 * np.sin(n / 40.0) + 0.3 * np.cos(e / 55.0), 0.5 * np.cos((n + e) / 50.0)], -1)


def _flights(seed, n_flights=6, places=8, flight_sd=0.0):
    rng = np.random.default_rng(seed)
    errs, poss = [], []
    for k in range(n_flights):
        common = rng.normal(0.0, flight_sd, 2)
        e, p = [], []
        for _ in range(places):
            c = rng.uniform(-150, 150, 2)
            c = np.floor(c / 20.0) * 20.0 + 10.0             # centre of a 20 m place
            pts = c + rng.uniform(-6, 6, (25, 2))
            e.append(_smooth_field(pts[:, 0], pts[:, 1]) + common + rng.normal(0.0, 0.1, (25, 2)))
            p.append(pts)
        errs.append(np.concatenate(e)); poss.append(np.concatenate(p))
    return errs, poss


def test_variogram_recovers_a_spatially_correlated_offset():
    errs, poss = _flights(0, n_flights=12)
    c, o, _, w = place_offsets(errs, poss)
    nug, ps, rng_m = fit_variogram(c, o)
    assert nug < 0.05 and 10.0 < rng_m < 300.0 and ps > 0.05
    assert w == pytest.approx(0.1 ** 2, rel=0.3)


def test_field_predicts_the_offset_near_places_and_falls_back_far_away():
    errs, poss = _flights(1, n_flights=10)
    f = fit_field(errs, poss)
    near = np.array([[c[0] + 2.0, c[1] - 2.0] for c in f.centres[:5]])
    m, v = f.predict(near)
    truth = _smooth_field(near[:, 0], near[:, 1])
    assert np.max(np.abs(m - truth)) < 0.25
    far_m, far_v = f.predict([[5000.0, 5000.0]])
    assert far_v[0] > 5 * np.max(v)                       # sill far from the data
    assert np.allclose(far_m[0], f.offsets.mean(axis=0), atol=0.3)


def test_flight_variance_appears_only_with_a_common_flight_offset():
    a = fit_field(*_flights(2, n_flights=12, flight_sd=0.0))
    b = fit_field(*_flights(2, n_flights=12, flight_sd=0.5))      # true flight variance 0.25 m^2
    assert a.flight_variance_m2 < 0.05 and 0.08 < b.flight_variance_m2 < 0.6


def test_held_out_flights_are_consistent_and_json_round_trips():
    errs, poss = _flights(3, n_flights=7)
    cv = cross_validated_field_nees(errs, poss, fit_flights=5)
    # Conservative on a smooth field (kriging variance of a fitted variogram).
    assert cv["holdouts"] == 42 and 0.3 < cv["median"] < 3.5
    f = fit_field(errs[:5], poss[:5])
    g = MapErrorField.from_json(f.to_json())
    assert np.allclose(g.predict(poss[6][:10])[0], f.predict(poss[6][:10])[0], atol=1e-3)
    assert field_nees(g, errs[6], poss[6]).shape == (len(errs[6]),)
    with pytest.raises(ValueError):
        cross_validated_field_nees(errs, poss, fit_flights=7)


def _trend_flights(seed, n_flights=8, places=8, north_max=60.0):
    """Offset 1.5 cm/m in N plus the smooth field; places only up to north_max."""
    rng = np.random.default_rng(seed)
    errs, poss = [], []
    for _ in range(n_flights):
        e, p = [], []
        for _ in range(places):
            c = np.array([rng.uniform(-150, north_max), rng.uniform(-150, 150)])
            pts = c + rng.uniform(-6, 6, (25, 2))
            off = _smooth_field(pts[:, 0], pts[:, 1]) * 0.5 + np.column_stack([0.015 * pts[:, 0], 0.0 * pts[:, 1]])
            e.append(off + rng.normal(0.0, 0.1, (25, 2)))
            p.append(pts)
        errs.append(np.concatenate(e)); poss.append(np.concatenate(p))
    return errs, poss


def test_linear_drift_extrapolates_a_trend():
    errs, poss = _trend_flights(4)
    const, lin = fit_field(errs, poss), fit_field(errs, poss, drift="linear")
    far = np.array([[150.0, 0.0], [140.0, 50.0]])                 # 80-90 m beyond the northmost place
    truth = 0.5 * _smooth_field(far[:, 0], far[:, 1]) + np.column_stack([0.015 * far[:, 0], np.zeros(2)])
    err_c = np.linalg.norm(const.predict(far)[0] - truth, axis=1)
    err_l = np.linalg.norm(lin.predict(far)[0] - truth, axis=1)
    assert err_l.max() < 0.6 and err_l.mean() < err_c.mean()
    # Far-away NEES of the linear field stays plausible (< 3 per axis pair on average).
    assert np.mean(np.sum((lin.predict(far)[0] - truth) ** 2, 1) / lin.fix_variance(far)) < 6.0
    g = MapErrorField.from_json(lin.to_json())
    assert g.drift == "linear" and np.allclose(g.predict(far)[0], lin.predict(far)[0], atol=1e-3)


def test_linear_drift_reduces_to_ordinary_on_flat_offsets():
    errs, poss = _flights(5, n_flights=8)
    lin = fit_field(errs, poss, drift="linear")
    near = np.array([[0.0, 0.0], [50.0, -40.0]])
    assert np.all(lin.fix_variance(near) < 1.0)
    with pytest.raises(ValueError):
        MapErrorField(np.zeros((3, 2)), np.zeros((3, 2)), 0.0, 1.0, 50.0, 0.0, drift="quadratic")


def test_linear_drift_rejects_collinear_places():
    rng = np.random.default_rng(6)
    errs, poss = [], []
    for _ in range(3):
        n = rng.uniform(-100, 100, 8)
        pts = np.concatenate([np.column_stack([np.full(25, x), np.full(25, 0.0)]) + rng.uniform(-3, 3, (25, 2))
                              for x in np.floor(n / 20.0) * 20.0 + 10.0])
        pts[:, 1] = 10.0                                         # one straight leg: E constant
        errs.append(rng.normal(0.0, 0.2, (len(pts), 2))); poss.append(pts)
    with pytest.raises(ValueError):
        fit_field(errs, poss, drift="linear")
    fit_field(errs, poss)                                        # constant mean is fine on a line
