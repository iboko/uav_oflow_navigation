"""Separate-flight error and covariance calibration for orthophoto fixes.

Training flights estimate a common 2D bias and a covariance of residuals
around that bias. Held-out flights are NEVER used to fit either quantity.
All observations must be time-matched to independent ground truth in the
same local N/E datum. Repeated frames are correlated: flights are the
sampling unit for weighting; this is not a certification study.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Sequence

import numpy as np

from .map_covariance_intersection import _covariance


@dataclass(frozen=True)
class MapErrorFlight:
    name: str
    # Each row: [N_map-N_ref, E_map-E_ref], metres.
    errors_ne_m: tuple[tuple[float, float], ...]
    total_frames: int
    false_fixes: int


@dataclass(frozen=True)
class MapErrorCalibration:
    bias_ne_m: tuple[float, float]
    covariance_ne_m2: tuple[tuple[float, float], tuple[float, float]]
    fit_flights: tuple[str, ...]
    holdout_flights: tuple[str, ...]
    holdout_nees_coverage_95pct: float
    holdout_mean_nees_2d: float
    holdout_false_fix_rate_among_accepted: float
    holdout_flight_details: dict
    independently_validated_for_flight: bool = False
    # Decomposition of covariance_ne_m2 (= within + between): scatter of
    # fixes around their own flight mean, and scatter of the flight means
    # around the pooled bias (a per-flight systematic error, e.g. the tilt
    # error of the attitude used for ortho-rectification).
    within_flight_covariance_ne_m2: tuple[tuple[float, float], tuple[float, float]] | None = None
    between_flight_covariance_ne_m2: tuple[tuple[float, float], tuple[float, float]] | None = None


def _errors(flight: MapErrorFlight, *, min_samples: int) -> np.ndarray:
    if not flight.name.strip():
        raise ValueError("Пустой идентификатор полета")
    try:
        errors = np.asarray(flight.errors_ne_m, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Некорректные ошибки координат") from exc
    if errors.ndim != 2 or errors.shape[1] != 2 or not np.isfinite(errors).all():
        raise ValueError("Ошибки координат должны быть конечными двумерными значениями")
    if len(errors) < min_samples:
        raise ValueError("Недостаточно измерений на независимом полете")
    if flight.total_frames < len(errors) or not 0 <= flight.false_fixes <= len(errors):
        raise ValueError("Неверная полнота данных и число ложных привязок")
    return errors


def calibrate_map_error(
    fitting_flights: Sequence[MapErrorFlight],
    held_out_flights: Sequence[MapErrorFlight],
    *,
    min_fitting_flights: int = 3,
    min_holdout_flights: int = 2,
    min_samples_per_flight: int = 30,
    outliers_removed_by_rule: str | None = None,
) -> MapErrorCalibration:
    """Estimate bias and *centered* 2D covariance, then report holdout coverage.

    Equal weight to each training flight despite different frame rates.
    Numerical covariance is NOT by itself an indication of certified
    statistical accuracy. No invented regularization/floor: singular
    empirical covariance raises, calling for better flight excitation.
    """
    if not 2 <= min_fitting_flights <= 100 or not 1 <= min_holdout_flights <= 100:
        raise ValueError("Недопустимое требуемое число полетов")
    if not 10 <= min_samples_per_flight <= 100000:
        raise ValueError("Недопустимое минимальное число кадров на полет")
    fit = list(fitting_flights)
    holdout = list(held_out_flights)
    if len(fit) < min_fitting_flights or len(holdout) < min_holdout_flights:
        raise ValueError("Недостаточно независимых полетов для разделения данных")
    labels = [x.name for x in fit + holdout]
    if len(set(labels)) != len(labels):
        raise ValueError("Полеты настройки и независимой проверки пересекаются")
    if any(f.false_fixes for f in fit) and not outliers_removed_by_rule:
        raise ValueError(
            "В обучающих данных есть ложные принятые привязки; "
            "одна гауссова ковариация непригодна без модели выбросов"
        )
    # With an explicit outlier rule the errors_ne_m are the inliers by that
    # rule; false_fixes counts the removed ones. Their rate is reported (also
    # for the fit flights): the onboard gate has to reject them, the
    # Gaussian covariance does not describe them.
    train = [_errors(f, min_samples=min_samples_per_flight) for f in fit]
    test = [_errors(f, min_samples=min_samples_per_flight) for f in holdout]
    bias = np.mean(np.stack([np.mean(x, axis=0) for x in train]), axis=0)
    # Equal flight weighting, preserving inter-flight drift around pooled bias.
    cov = np.mean(np.stack([
        (x-bias).T @ (x-bias) / len(x) for x in train
    ]), axis=0)
    cov = _covariance(cov, "оцененная ковариация ошибок карты")
    means = np.stack([np.mean(x, axis=0) for x in train])
    between = np.mean(np.stack([np.outer(m - bias, m - bias) for m in means]), axis=0)
    within_flight = cov - between

    details = {}
    flight_coverages = []
    flight_means = []
    flight_false_rates = []
    for flight, errors in zip(holdout, test):
        corrected = errors - bias
        nees = np.einsum("ij,ij->i", corrected,
                         np.linalg.solve(cov, corrected.T).T)
        if not np.isfinite(nees).all() or np.min(nees) < -1e-10:
            raise ValueError("Недопустимые нормированные ошибки на эталоне")
        within = nees <= 5.991464547107979
        flight_coverages.append(float(np.mean(within)))
        flight_means.append(float(np.mean(nees)))
        flight_false_rates.append(flight.false_fixes / len(errors))
        details[flight.name] = {
            "matched_accepted_frames": len(errors),
            "total_frames": flight.total_frames,
            "false_fixes": flight.false_fixes,
            "mean_nees_2d": float(np.mean(nees)),
            "fraction_within_chi2_2_95": float(np.mean(within)),
            "bias_corrected_position_rmse_m": sqrt(
                float(np.mean(np.sum(corrected**2, axis=1)))
            ),
        }
    return MapErrorCalibration(
        tuple(float(x) for x in bias),
        tuple(tuple(float(v) for v in line) for line in cov),
        tuple(x.name for x in fit),
        tuple(x.name for x in holdout),
        float(np.mean(flight_coverages)),
        float(np.mean(flight_means)),
        float(np.mean(flight_false_rates)),
        details,
        False,
        tuple(tuple(float(v) for v in line) for line in within_flight),
        tuple(tuple(float(v) for v in line) for line in between),
    )


def covariance_at_height(calibration: dict, height_m: float) -> np.ndarray:
    """Fix covariance at the frame height from a calibration JSON.

    Without the decomposition: the calibrated covariance as is. With it:
    within + between * (h / h_cal)^2. The between-flight part is taken as
    a tilt error of the attitude used for ortho-rectification (the
    nadir point moves by h * tilt; SWISSIMAGE series, ulogs). The
    within-flight part (correlation noise, map vs scene) is kept as
    calibrated. A model assumption: flights at other heights must check it.
    """
    total = np.asarray(calibration["covariance_ne_m2"], dtype=float)
    within = calibration.get("within_flight_covariance_ne_m2")
    between = calibration.get("between_flight_covariance_ne_m2")
    h_cal = calibration.get("calibration_height_m")
    if within is None or between is None or h_cal is None:
        return total
    if not (isfinite(height_m) and height_m > 0 and isfinite(h_cal) and h_cal > 0):
        raise ValueError("Высота привязки и калибровки должна быть положительной")
    return _covariance(np.asarray(within, dtype=float) + np.asarray(between, dtype=float)
                       * (height_m / h_cal) ** 2, "ковариация привязки на высоте")


def quality_covariance(floor_m2: float, k2: float, subpatch_cov_ne_m2) -> np.ndarray:
    """Fix covariance from the match quality: floor * I + k2 * C_subpatch."""
    c = np.asarray(subpatch_cov_ne_m2, dtype=float)
    return floor_m2 * np.eye(2) + k2 * 0.5 * (c + c.T)


def fit_quality_model(fit_errors: Sequence[np.ndarray], fit_subcov: Sequence[np.ndarray],
                      holdout_errors: Sequence[np.ndarray], holdout_subcov: Sequence[np.ndarray],
                      bias_ne_m) -> dict:
    """Maximum likelihood (floor, k2) of R_i = floor*I + k2*C_i on the fit flights.

    errors: per flight (n, 2) residuals [m] (bias is subtracted here);
    subcov: per flight (n, 2, 2) sub-patch covariances of the same fixes.
    Each flight has equal weight (mean log-likelihood per flight). Grid
    search on a log grid: two parameters, no optimiser dependency. The
    held-out flights are only evaluated (NEES, 95 % coverage).
    """
    b = np.asarray(bias_ne_m, dtype=float)
    fit = [(np.asarray(e, float) - b, np.asarray(c, float)) for e, c in zip(fit_errors, fit_subcov)]
    hold = [(np.asarray(e, float) - b, np.asarray(c, float)) for e, c in zip(holdout_errors, holdout_subcov)]
    if not fit or any(len(e) != len(c) or len(e) < 10 for e, c in fit + hold):
        raise ValueError("Нужны ошибки и ковариации четвертей для каждой привязки")

    def terms(e, c, floor, k2):
        cs = 0.5 * (c + np.transpose(c, (0, 2, 1)))
        r = floor * np.eye(2)[None] + k2 * cs
        det = r[:, 0, 0] * r[:, 1, 1] - r[:, 0, 1] ** 2
        inv = np.stack([np.stack([r[:, 1, 1], -r[:, 0, 1]], -1), np.stack([-r[:, 1, 0], r[:, 0, 0]], -1)], -2) \
            / det[:, None, None]
        nees = np.einsum("ni,nij,nj->n", e, inv, e)
        return np.log(det), nees

    best = None
    for floor in np.exp(np.linspace(np.log(1e-3), np.log(4.0), 60)):
        for k2 in np.concatenate([[0.0], np.exp(np.linspace(np.log(1e-3), np.log(20.0), 70))]):
            nll = 0.0
            for e, c in fit:
                ld, nees = terms(e, c, floor, k2)
                nll += float(np.mean(ld + nees))
            if best is None or nll < best[0]:
                best = (nll, floor, k2)
    _, floor, k2 = best
    per = []
    for e, c in hold:
        _, nees = terms(e, c, floor, k2)
        per.append(nees)
    return {"floor_m2": float(floor), "k2": float(k2),
            "fit_mean_nees": [round(float(np.mean(terms(e, c, floor, k2)[1])), 3) for e, c in fit],
            "holdout_mean_nees": [round(float(np.mean(n)), 3) for n in per],
            "holdout_coverage_95": [round(float(np.mean(n <= 5.991464547107979)), 3) for n in per],
            "model": "R = floor_m2 * I + k2 * C_subpatch"}


def place_outlier_rule(errors_ne_m, positions_ne_m, *, cell_m: float = 20.0, false_place_m: float = 3.0,
                       false_fix_m: float = 1.0, min_fixes_per_place: int = 5) -> dict:
    """Split one flight's fix errors into inliers, false fixes and false places.

    The map error is a per-place offset plus small noise (series H: 36
    places of 20 m, within-place SD 0.2 m, place offsets mostly < 2 m and
    3 places 4-21 m off with self-consistent fixes). A flight-wide median
    rule mixes those. Here: places = cells of ``cell_m`` along the
    flown positions; a place whose median error is more than
    ``false_place_m`` from the flight's median of place medians is a
    false place (all its fixes are counted, not fitted); within a place a
    fix more than ``false_fix_m`` from the place median is a false fix.
    Places with fewer than ``min_fixes_per_place`` fixes are left out.
    Thresholds are fixed before use; they are not fitted.
    """
    e = np.asarray(errors_ne_m, dtype=float)
    p = np.asarray(positions_ne_m, dtype=float)
    if e.shape != p.shape or e.ndim != 2 or e.shape[1] != 2:
        raise ValueError("Ошибки и положения должны совпадать по форме (n, 2)")
    key = np.floor(p / cell_m).astype(int)
    cells = {}
    for i, k in enumerate(map(tuple, key)):
        cells.setdefault(k, []).append(i)
    places = {k: np.asarray(v) for k, v in cells.items() if len(v) >= min_fixes_per_place}
    inlier = np.zeros(len(e), bool)
    false_fix = np.zeros(len(e), bool)
    false_place = np.zeros(len(e), bool)
    if not places:
        return {"inlier": inlier, "false_fix": false_fix, "false_place": false_place,
                "places": 0, "false_places": 0}
    medians = {k: np.median(e[idx], axis=0) for k, idx in places.items()}
    centre = np.median(np.stack(list(medians.values())), axis=0)
    n_false_places = 0
    for k, idx in places.items():
        if np.linalg.norm(medians[k] - centre) > false_place_m:
            false_place[idx] = True
            n_false_places += 1
            continue
        far = np.linalg.norm(e[idx] - medians[k], axis=1) > false_fix_m
        false_fix[idx[far]] = True
        inlier[idx[~far]] = True
    return {"inlier": inlier, "false_fix": false_fix, "false_place": false_place,
            "places": len(places), "false_places": n_false_places}
