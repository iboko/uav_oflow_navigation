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
    train = [_errors(f, min_samples=min_samples_per_flight) for f in fit]
    test = [_errors(f, min_samples=min_samples_per_flight) for f in holdout]
    bias = np.mean(np.stack([np.mean(x, axis=0) for x in train]), axis=0)
    # Equal flight weighting, preserving inter-flight drift around pooled bias.
    cov = np.mean(np.stack([
        (x-bias).T @ (x-bias) / len(x) for x in train
    ]), axis=0)
    cov = _covariance(cov, "оцененная ковариация ошибок карты")

    details = {}
    accepted_total = 0
    rejected_total = 0
    covered = []
    all_nees = []
    for flight, errors in zip(holdout, test):
        corrected = errors - bias
        nees = np.einsum("ij,ij->i", corrected,
                         np.linalg.solve(cov, corrected.T).T)
        if not np.isfinite(nees).all() or np.min(nees) < -1e-10:
            raise ValueError("Недопустимые нормированные ошибки на эталоне")
        within = nees <= 5.991464547107979
        covered.extend(bool(v) for v in within)
        all_nees.extend(float(x) for x in nees)
        accepted_total += len(errors)
        rejected_total += flight.false_fixes
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
        sum(covered) / len(covered),
        float(np.mean(all_nees)),
        rejected_total / accepted_total,
        details,
        False,
    )
