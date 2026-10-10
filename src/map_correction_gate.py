"""Non-actuating safety gate for orthophoto position candidates.

Reject map fixes without independent calibration of their 2x2 covariance
and without justified cross-correlation model. This module intentionally
does NOT inject a position measurement into PX4 or the 15-state ESKF.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

from .orthophoto_localization import MapLocalizationFix


@dataclass(frozen=True)
class MapCorrectionAssessment:
    eligible: bool
    status: str
    delta_north_m: float = float("nan")
    delta_east_m: float = float("nan")
    nis_2d: float = float("nan")


def assess_map_correction(
    fix: MapLocalizationFix,
    *,
    predicted_ne_m: tuple[float, float],
    predicted_covariance_ne_m2: tuple[tuple[float, float], tuple[float, float]],
    map_covariance_ne_m2: tuple[tuple[float, float], tuple[float, float]] | None,
    map_error_statistics_validated: bool,
    measurement_errors_independent_from_prior: bool,
    max_innovation_distance_m: float = 20.,
    nis_limit: float = 9.210340371976184,
) -> MapCorrectionAssessment:
    """Check possible correction, never mutate a flight-navigation state."""
    def reject(reason: str) -> MapCorrectionAssessment:
        return MapCorrectionAssessment(False, reason)
    if not fix.accepted:
        return reject("MAP_FIX_REJECTED")
    if not map_error_statistics_validated or map_covariance_ne_m2 is None:
        return reject("MAP_COVARIANCE_UNVALIDATED")
    if not measurement_errors_independent_from_prior:
        # Usually violated if both the localizer and VO use the same frames.
        # ESKF fusion then needs cross-covariance or another robust method.
        return reject("VISUAL_CROSS_CORRELATION_UNMODELED")
    try:
        pred = np.asarray(predicted_ne_m, dtype=float)
        observed = np.array((fix.north_m, fix.east_m), dtype=float)
        pp = np.asarray(predicted_covariance_ne_m2, dtype=float)
        rr = np.asarray(map_covariance_ne_m2, dtype=float)
        if not all(np.isfinite(x).all() for x in (pred, observed, pp, rr)):
            return reject("NONFINITE_POSITION_OR_COVARIANCE")
        if pred.shape != (2,) or pp.shape != (2, 2) or rr.shape != (2, 2):
            return reject("INVALID_POSITION_OR_COVARIANCE_SHAPE")
        if (not np.allclose(pp, pp.T, atol=1e-8) or
                not np.allclose(rr, rr.T, atol=1e-8)):
            return reject("ASYMMETRIC_COVARIANCE")
        np.linalg.cholesky(pp)
        np.linalg.cholesky(rr)
        residual = observed - pred
        if not isfinite(max_innovation_distance_m) or max_innovation_distance_m <= 0:
            return reject("INVALID_GATE_CONFIGURATION")
        if float(np.linalg.norm(residual)) > max_innovation_distance_m:
            return reject("LARGE_MAP_JUMP")
        ss = pp + rr
        whitened = np.linalg.solve(np.linalg.cholesky(ss), residual)
        nis = float(whitened @ whitened)
        if not isfinite(nis_limit) or nis_limit <= 0:
            return reject("INVALID_GATE_CONFIGURATION")
        if nis > nis_limit:
            return MapCorrectionAssessment(
                False, "MAP_INNOVATION_REJECTED",
                float(residual[0]), float(residual[1]), nis
            )
        return MapCorrectionAssessment(
            True, "CANDIDATE_ONLY_NOT_APPLIED",
            float(residual[0]), float(residual[1]), nis
        )
    except (TypeError, ValueError, np.linalg.LinAlgError):
        return reject("INVALID_COVARIANCE")
