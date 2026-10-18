"""Map error as a spatial field: kriged place offsets.

Series M/N (3D world, 16 flights): the map offset is a property of the
place, not of the flight. Places within 5 m seen on different flights
differ by 0.10 m (semivariance per axis 0.010 m^2); the difference grows
to 0.58 m at 30-60 m and levels at 0.8 m beyond 120 m. A constant bias
with one covariance therefore either misses the regional offset or has
to be wide everywhere.

Here: place offsets (mean fix error per 20 m place) from calibration
flights; an exponential semivariogram per axis (shared parameters) fitted
by weighted least squares to the binned pairs; ordinary kriging gives the
expected offset and its variance at any point. A fix covariance is

    R = (kriging variance + within-place variance + flight variance) * I

The flight variance is the excess of a left-out flight's mean residual
over what the field predicts (leave one fit flight out): some flights
carry a common offset (series N, flight 1: 1 m in E at a place 4 m from
places of other flights). Far from the calibration places the kriging
variance tends to the sill: the field is no better than a constant there.

Optional linear drift (universal kriging, ``drift="linear"``): the mean
offset is a + b N + c E instead of a constant; the variogram is fitted to
the residuals of the least-squares trend. Tried because place offsets grow
to the north (series M/N/P, ~1.4 cm/m; the 2019 map against the 2025 scene
texture shows the same trend, so it is in the imagery) and the constant-mean
variogram range runs to its 500-m grid limit. Rejected on the flights: the
residual range drops to ~20 m, the share of held-out flights with NEES > 3
grows (0.10-0.12 -> 0.18) and series P flight 1, 90 m from the nearest
calibration place, stays at NEES ~7. Kept for reproducing that comparison;
the default stays constant.

Isotropic, stationary model; short-scale features (roof edges) are not
resolved below the place size. Not a certified error model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .map_error_calibration import place_groups

DRIFTS = ("constant", "linear")
_BINS_M = (0.0, 5.0, 10.0, 20.0, 30.0, 45.0, 60.0, 90.0, 120.0, 200.0, 400.0)


def place_offsets(errors_by_flight: Sequence, positions_by_flight: Sequence, *, cell_m: float = 20.0,
                  min_fixes_per_place: int = 5):
    """Place centres, mean offsets, flight index and the pooled within-place variance per axis."""
    centres, offsets, flight, within = [], [], [], []
    for k, (e, p) in enumerate(zip(errors_by_flight, positions_by_flight)):
        e, p = np.asarray(e, float), np.asarray(p, float)
        for idx in place_groups(p, cell_m=cell_m, min_fixes_per_place=min_fixes_per_place):
            centres.append(p[idx].mean(axis=0))
            offsets.append(e[idx].mean(axis=0))
            flight.append(k)
            within.append(float(np.mean(np.sum((e[idx] - offsets[-1]) ** 2, axis=1))) / 2)
    if len(centres) < 3:
        raise ValueError("Недостаточно мест для поля ошибки карты")
    return np.asarray(centres), np.asarray(offsets), np.asarray(flight), float(np.mean(within))


def fit_variogram(centres: np.ndarray, offsets: np.ndarray, *, min_pairs: int = 5) -> tuple[float, float, float]:
    """(nugget, partial sill, range) of gamma(h) = c0 + c1 (1 - exp(-h / a)) per axis.

    Range on a log grid, nugget and sill by non-negative weighted least
    squares for each range (weights: pairs per bin)."""
    d = np.linalg.norm(centres[:, None] - centres[None], axis=2)
    g = 0.25 * np.sum((offsets[:, None] - offsets[None]) ** 2, axis=2)
    iu = np.triu_indices(len(centres), 1)
    d, g = d[iu], g[iu]
    h, gm, nb = [], [], []
    for lo, hi in zip(_BINS_M[:-1], _BINS_M[1:]):
        s = (d >= lo) & (d < hi)
        if s.sum() >= min_pairs:
            h.append(d[s].mean()); gm.append(g[s].mean()); nb.append(s.sum())
    if len(h) < 3:
        raise ValueError("Недостаточно пар мест для вариограммы")
    h, gm, w = np.asarray(h), np.asarray(gm), np.sqrt(np.asarray(nb, float))
    best = None
    for a in np.geomspace(2.0, 500.0, 120):
        basis = np.column_stack([np.ones_like(h), 1.0 - np.exp(-h / a)])
        c, *_ = np.linalg.lstsq(basis * w[:, None], gm * w, rcond=None)
        c = np.maximum(c, (0.0, 1e-4))
        r = float(np.sum((w * (basis @ c - gm)) ** 2))
        if best is None or r < best[0]:
            best = (r, float(c[0]), float(c[1]), float(a))
    return best[1], best[2], best[3]


@dataclass
class MapErrorField:
    centres: np.ndarray            # (n, 2) place centres, local N/E
    offsets: np.ndarray            # (n, 2) mean fix error per place
    nugget: float
    partial_sill: float
    range_m: float
    within_variance_m2: float
    flight_variance_m2: float = 0.0
    drift: str = "constant"

    def __post_init__(self) -> None:
        if self.drift not in DRIFTS:
            raise ValueError(f"drift: {DRIFTS}")
        self.centres = np.asarray(self.centres, float)
        self.offsets = np.asarray(self.offsets, float)
        n = len(self.centres)
        k = self._cov(np.linalg.norm(self.centres[:, None] - self.centres[None], axis=2)) + self.nugget * np.eye(n)
        f = self._basis(self.centres)
        self._a = np.block([[k, f], [f.T, np.zeros((f.shape[1], f.shape[1]))]])
        self._a_inv = np.linalg.pinv(self._a)

    def _basis(self, q: np.ndarray) -> np.ndarray:
        """Drift basis at points: 1, or 1, N, E (in 100 m, for conditioning)."""
        q = np.atleast_2d(q)
        return np.ones((len(q), 1)) if self.drift == "constant" else np.column_stack([np.ones(len(q)), q / 100.0])

    def _cov(self, d):
        return self.partial_sill * np.exp(-np.asarray(d) / self.range_m)

    def predict(self, points_ne_m) -> tuple[np.ndarray, np.ndarray]:
        """Expected offset (n, 2) and kriging variance per axis (n,) at the points."""
        q = np.atleast_2d(np.asarray(points_ne_m, float))
        k = self._cov(np.linalg.norm(q[:, None] - self.centres[None], axis=2))
        f = self._basis(q)
        n = len(self.centres)
        sol = self._a_inv @ np.vstack([k.T, f.T])
        w, mu = sol[:n], sol[n:]
        mean = w.T @ self.offsets
        var = self.partial_sill + self.nugget - np.sum(w * k.T, axis=0) - np.sum(mu * f.T, axis=0)
        return mean, np.maximum(var, 0.0)

    def fix_variance(self, points_ne_m) -> np.ndarray:
        """Variance per axis of a single fix error around the predicted offset."""
        return self.predict(points_ne_m)[1] + self.within_variance_m2 + self.flight_variance_m2

    def to_json(self) -> dict:
        return {"centres_ne_m": self.centres.round(3).tolist(), "offsets_ne_m": self.offsets.round(4).tolist(),
                "nugget_m2": self.nugget, "partial_sill_m2": self.partial_sill, "range_m": self.range_m,
                "within_variance_m2": self.within_variance_m2, "flight_variance_m2": self.flight_variance_m2,
                "drift": self.drift}

    @classmethod
    def from_json(cls, d: dict) -> "MapErrorField":
        return cls(np.asarray(d["centres_ne_m"]), np.asarray(d["offsets_ne_m"]), d["nugget_m2"],
                   d["partial_sill_m2"], d["range_m"], d["within_variance_m2"], d.get("flight_variance_m2", 0.0),
                   d.get("drift", "constant"))


def detrend(centres: np.ndarray, offsets: np.ndarray, drift: str) -> np.ndarray:
    """Offsets minus their least-squares drift (the variogram is fitted to these)."""
    if drift == "constant":
        return offsets - offsets.mean(axis=0)
    f = np.column_stack([np.ones(len(centres)), centres / 100.0])
    return offsets - f @ np.linalg.lstsq(f, offsets, rcond=None)[0]


def drift_identifiable(centres: np.ndarray, drift: str) -> bool:
    """A linear drift needs places spanning both axes (not all on one line)."""
    if drift == "constant":
        return len(centres) >= 3
    f = np.column_stack([np.ones(len(centres)), np.asarray(centres) / 100.0])
    return len(centres) >= 6 and np.linalg.matrix_rank(f, tol=1e-3) == 3


def fit_field(errors_by_flight: Sequence, positions_by_flight: Sequence, *, flight_variance: bool = True,
              cell_m: float = 20.0, drift: str = "constant") -> MapErrorField:
    """Field from calibration flights; flight variance by leaving each flight out."""
    c, o, fl, w = place_offsets(errors_by_flight, positions_by_flight, cell_m=cell_m)
    if drift == "linear" and not drift_identifiable(c, drift):
        raise ValueError("Линейный тренд не определяется: меньше 6 мест или все места на одной прямой")
    field = MapErrorField(c, o, *fit_variogram(c, detrend(c, o, drift)), w, drift=drift)
    if flight_variance and len(errors_by_flight) >= 3:
        excess = []
        for k in range(len(errors_by_flight)):
            keep = fl != k
            if not drift_identifiable(c[keep], drift):
                continue
            sub = MapErrorField(c[keep], o[keep], field.nugget, field.partial_sill, field.range_m, w, drift=drift)
            e, p = np.asarray(errors_by_flight[k], float), np.asarray(positions_by_flight[k], float)
            m, v = sub.predict(p)
            n_places = max(1, int(np.sum(fl == k)))
            # Mean residual of the flight against the field of the others,
            # minus the part explained by the field variance itself.
            excess.append(float(np.sum((e - m).mean(axis=0) ** 2)) / 2 - float(np.mean(v + w)) / n_places)
        field.flight_variance_m2 = max(0.0, float(np.mean(excess))) if excess else 0.0
    return field


def field_nees(field: MapErrorField, errors, positions) -> np.ndarray:
    """2D NEES of each fix against the field."""
    e, p = np.asarray(errors, float), np.asarray(positions, float)
    m, _ = field.predict(p)
    return np.sum((e - m) ** 2, axis=1) / field.fix_variance(p)


def cross_validated_field_nees(errors_by_flight: Sequence, positions_by_flight: Sequence, *,
                               fit_flights: int, drift: str = "constant") -> dict:
    """Mean NEES of each held-out flight over every split (fit_flights fitted)."""
    from itertools import combinations
    n = len(errors_by_flight)
    if not 2 <= fit_flights < n:
        raise ValueError("Недопустимое число обучающих полетов")
    out = []
    for fit in combinations(range(n), fit_flights):
        f = fit_field([errors_by_flight[i] for i in fit], [positions_by_flight[i] for i in fit], drift=drift)
        for j in (j for j in range(n) if j not in fit):
            out.append(float(np.mean(field_nees(f, errors_by_flight[j], positions_by_flight[j]))))
    x = np.asarray(out)
    return {"median": float(np.median(x)), "mean": float(x.mean()), "quantile_90": float(np.quantile(x, 0.9)),
            "quantile_95": float(np.quantile(x, 0.95)), "fraction_above_3": float(np.mean(x > 3.0)),
            "holdouts": len(x), "per_holdout": [round(float(v), 3) for v in x]}
