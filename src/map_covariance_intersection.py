"""Conservative 2D position fusion with UNKNOWN cross-correlation.

Covariance intersection (CI) avoids silently assuming independence between
orthophoto matching and visual-inertial odometry when they use shared images.
Both marginal covariances must separately be credible, SPD and expressed
in the SAME reference frame/time. This is an offline hypothetical posterior;
not the correct 15x15 ESKF update, not an integrity or false-match proof.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
import numpy as np

from .orthophoto_localization import MapLocalizationFix


def _vector(a: object, n: int, label: str) -> np.ndarray:
    try:
        r = np.asarray(a, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Неверный вектор: {label}") from exc
    if r.shape != (n,) or not np.isfinite(r).all():
        raise ValueError(f"Неконечный или неверной размерности вектор: {label}")
    return r


def _covariance(a: object, label: str) -> np.ndarray:
    try:
        p = np.asarray(a, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Неверная ковариация: {label}") from exc
    if p.shape != (2, 2) or not np.isfinite(p).all():
        raise ValueError(f"Неконечная или неверной размерности ковариация: {label}")
    if not np.allclose(p, p.T, rtol=1e-10, atol=1e-12):
        raise ValueError(f"Несимметричная ковариация: {label}")
    try:
        np.linalg.cholesky(p)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"Ковариация не положительно определена: {label}") from exc
    return 0.5 * (p + p.T)


def covariance_intersection_2d(
    prior_ne_m: object, prior_covariance_ne_m2: object,
    map_ne_m: object, map_covariance_ne_m2: object,
    *, grid_steps: int = 201,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Minimize log(det(PCI)) over w in [0,1] by deterministic grid.

    PCI^-1 = w Pprior^-1 + (1-w) Pmap^-1
    xCI = PCI (w Pprior^-1 xprior + (1-w) Pmap^-1 xmap)
    No additive confidence gain for identical fully correlated estimates.
    """
    if not isinstance(grid_steps, int) or not 21 <= grid_steps <= 2001:
        raise ValueError("Неверное число шагов оптимизации")
    x1 = _vector(prior_ne_m, 2, "предсказанное положение")
    x2 = _vector(map_ne_m, 2, "картографическое положение")
    p1 = _covariance(prior_covariance_ne_m2, "ковариация прогноза")
    p2 = _covariance(map_covariance_ne_m2, "ковариация карты")
    eye = np.eye(2)
    precision1 = np.linalg.solve(p1, eye)
    precision2 = np.linalg.solve(p2, eye)
    best = None
    for w in np.linspace(0., 1., grid_steps):
        combined_precision = w * precision1 + (1.-w) * precision2
        sign, logdet_precision = np.linalg.slogdet(combined_precision)
        if sign <= 0:
            raise ValueError("Вырожденная матрица объединения")
        if best is None or logdet_precision > best[0]:
            best = (logdet_precision, float(w), combined_precision)
    assert best is not None
    _, w, precision = best
    covariance = np.linalg.solve(precision, eye)
    covariance = 0.5 * (covariance + covariance.T)
    position = covariance @ (
        w * precision1 @ x1 + (1-w) * precision2 @ x2
    )
    if not np.isfinite(position).all() or not np.isfinite(covariance).all():
        raise ValueError("Численная ошибка объединения")
    return position, covariance, w


@dataclass(frozen=True)
class CorrelationAwareMapProposal:
    eligible: bool
    status: str
    position_ne_m: tuple[float, float] | None = None
    covariance_ne_m2: tuple[tuple[float, float], tuple[float, float]] | None = None
    weight_prior: float | None = None
    applied_to_eskf: bool = False


def propose_map_ci_correction(
    fix: MapLocalizationFix,
    *,
    prior_ne_m: tuple[float, float],
    prior_covariance_ne_m2: tuple[tuple[float, float], tuple[float, float]],
    map_covariance_ne_m2: tuple[tuple[float, float], tuple[float, float]] | None,
    prior_covariance_independently_validated: bool,
    map_covariance_independently_validated: bool,
    reference_frames_and_timestamps_aligned: bool,
    map_integrity_checked: bool,
    max_position_disagreement_m: float = 20.,
) -> CorrelationAwareMapProposal:
    """Evaluate the CI-only option; never update the full ESKF state.

    CI handles unknown mutual correlation *if each covariance is
    conservative*. It does not protect against consistently false map
    matches, bad calibration, different datums or a false integrity flag.
    """
    def reject(status: str) -> CorrelationAwareMapProposal:
        return CorrelationAwareMapProposal(False, status)
    if not fix.accepted:
        return reject("MAP_FIX_REJECTED")
    if not map_integrity_checked:
        return reject("MAP_INTEGRITY_NOT_ESTABLISHED")
    if not reference_frames_and_timestamps_aligned:
        return reject("MAP_FRAME_OR_TIME_UNALIGNED")
    if not prior_covariance_independently_validated:
        return reject("PRIOR_COVARIANCE_UNVALIDATED")
    if not map_covariance_independently_validated or map_covariance_ne_m2 is None:
        return reject("MAP_COVARIANCE_UNVALIDATED")
    if not isfinite(max_position_disagreement_m) or max_position_disagreement_m <= 0:
        return reject("INVALID_INNOVATION_LIMIT")
    try:
        x1 = _vector(prior_ne_m, 2, "прогноз")
        x2 = _vector((fix.north_m, fix.east_m), 2, "привязка")
        _covariance(prior_covariance_ne_m2, "прогноз")
        _covariance(map_covariance_ne_m2, "карта")
        if np.linalg.norm(x1-x2) > max_position_disagreement_m:
            return reject("MAP_POSITION_DISAGREEMENT")
        position, covariance, weight = covariance_intersection_2d(
            x1, prior_covariance_ne_m2, x2, map_covariance_ne_m2
        )
    except ValueError:
        return reject("INVALID_POSITION_OR_COVARIANCE")
    return CorrelationAwareMapProposal(
        True,
        "OFFLINE_CI_CANDIDATE_NOT_APPLIED",
        tuple(float(x) for x in position),
        tuple(tuple(float(v) for v in line) for line in covariance),
        float(weight),
        False,
    )
