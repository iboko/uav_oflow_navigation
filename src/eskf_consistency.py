"""Independent reference tests for GNSS-denied ESKF, offline only.

Never calibrate filter noise and validate it on the same reference flights.
The metrics below are screening diagnostics, not flightworthiness evidence.
Velocity NEES uses the 2x2 *marginal* velocity covariance, not its inverse
inside the full 15x15 covariance. Reference and estimate are exactly time
matched in the same NED frame and m/s units; no silent time interpolation.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import acos, isfinite, sqrt
from typing import Sequence

import numpy as np

from .error_state_inertial_filter import (
    EskfSnapshot, continuous_error_jacobian, exp_transition,
)
from .imu_preintegration import rotation_body_to_ned

CHI2_2_95 = 5.991464547107979
CHI2_2_99 = 9.210340371976184


@dataclass(frozen=True)
class TruthSample:
    timestamp_us: int
    velocity_ned_m_s: tuple[float, float, float]
    source: str
    # Optional: only for descriptive Euclidean trajectory errors.
    position_ned_m: tuple[float, float, float] | None = None
    quaternion_body_to_ned: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class LinearizationSample:
    dt_s: float
    rotation_body_to_ned: tuple[tuple[float, float, float], ...]
    omega_body_rad_s: tuple[float, float, float]
    specific_force_body_m_s2: tuple[float, float, float]


def _arr(value: object, shape: tuple[int, ...], label: str) -> np.ndarray:
    a = np.asarray(value, dtype=float)
    if a.shape != shape or not np.isfinite(a).all():
        raise ValueError(f"Неверный размер или неконечные данные: {label}")
    return a


def _spd(a: np.ndarray, label: str) -> None:
    if not np.allclose(a, a.T, atol=1e-9, rtol=1e-9):
        raise ValueError(f"Несимметричная ковариация: {label}")
    try:
        np.linalg.cholesky(a)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"Ковариация не положительно определена: {label}") from exc


def _check_reference(sample: TruthSample) -> None:
    if not isinstance(sample.timestamp_us, int) or sample.timestamp_us <= 0:
        raise ValueError("Недопустимое время независимого эталона")
    source = sample.source.strip().lower() if isinstance(sample.source, str) else ""
    if not source or source in ("filter", "estimator", "self", "synthetic_prediction"):
        raise ValueError("Не установлен независимый источник эталона")
    _arr(sample.velocity_ned_m_s, (3,), "эталонная скорость")


def nees_horizontal_velocity(
    snapshot: EskfSnapshot, truth: TruthSample,
) -> dict:
    """NEES of VN/VE (2 dof), exactly time-aligned independent truth."""
    _check_reference(truth)
    if snapshot.timestamp_us != truth.timestamp_us:
        raise ValueError("Не совпадают временные метки фильтра и эталона")
    estimated = _arr(snapshot.velocity_ned_m_s, (3,), "скорость РФК")
    reference = _arr(truth.velocity_ned_m_s, (3,), "эталонная скорость")
    cov = _arr(snapshot.covariance_15x15, (15, 15), "ковариация РФК")
    p = cov[3:5, 3:5]
    _spd(p, "горизонтальная скорость")
    error = estimated[:2] - reference[:2]
    metric = float(error @ np.linalg.solve(p, error))
    if not isfinite(metric) or metric < 0:
        raise ValueError("Недопустимый NEES")
    return {
        "timestamp_us": snapshot.timestamp_us,
        "reference_source": truth.source,
        "error_vn_m_s": float(error[0]),
        "error_ve_m_s": float(error[1]),
        "horizontal_speed_error_m_s": float(np.linalg.norm(error)),
        "nees_vne_2d": metric,
        "inside_95pct_chi2_2": metric <= CHI2_2_95,
    }


def quaternion_angular_error_rad(
    estimated_body_to_ned: Sequence[float],
    true_body_to_ned: Sequence[float],
) -> float:
    """Shortest SO(3) error angle; sign-invariant for unit quaternions."""
    q0 = _arr(estimated_body_to_ned, (4,), "кватернион РФК")
    q1 = _arr(true_body_to_ned, (4,), "кватернион эталона")
    n0, n1 = float(np.linalg.norm(q0)), float(np.linalg.norm(q1))
    if not 0.8 <= n0 <= 1.2 or not 0.8 <= n1 <= 1.2:
        raise ValueError("Некорректная норма кватерниона")
    dot = min(1.0, max(-1.0, abs(float(q0 @ q1 / (n0 * n1)))))
    return 2 * acos(dot)


def right_attitude_error_rotation_vector(
    estimated_body_to_ned: Sequence[float],
    true_body_to_ned: Sequence[float],
) -> np.ndarray:
    """Log(q_est^-1 * q_true) in previous/nominal body axes, shortest arc."""
    from math import atan2
    from .imu_preintegration import quat_product

    est = _arr(estimated_body_to_ned, (4,), "кватернион оценки")
    ref = _arr(true_body_to_ned, (4,), "кватернион эталона")
    norm_est, norm_ref = float(np.linalg.norm(est)), float(np.linalg.norm(ref))
    if not (0.8 <= norm_est <= 1.2 and 0.8 <= norm_ref <= 1.2):
        raise ValueError("Некорректная норма ориентации")
    est, ref = est / norm_est, ref / norm_ref
    relative = quat_product(
        np.array([est[0], -est[1], -est[2], -est[3]]), ref
    )
    if relative[0] < 0:
        relative = -relative
    vector = relative[1:]
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        return 2 * vector
    return (2 * atan2(norm, float(relative[0])) / norm) * vector


def nees_full_15(
    snapshot: EskfSnapshot,
    truth: TruthSample,
    *,
    truth_gyro_bias_rad_s: tuple[float, float, float],
    truth_accel_bias_m_s2: tuple[float, float, float],
    origin_alignment_verified: bool,
    heading_alignment_verified: bool,
) -> float:
    """15D NEES only with independently verified frame, origin and biases.

    The required "verified" booleans are declarations supplied by test staff,
    NOT proof of validation. Visual-only odometry lacks absolute origin/yaw.
    """
    if not origin_alignment_verified or not heading_alignment_verified:
        raise ValueError("Полный NEES запрещен без подтвержденной привязки начала и курса")
    _check_reference(truth)
    if snapshot.timestamp_us != truth.timestamp_us:
        raise ValueError("Метки времени полного NEES не совпадают")
    if truth.position_ned_m is None or truth.quaternion_body_to_ned is None:
        raise ValueError("Полный NEES требует независимых координат и ориентации")
    cov = _arr(snapshot.covariance_15x15, (15, 15), "полная ковариация")
    _spd(cov, "полная ковариация")
    error = np.concatenate((
        _arr(truth.position_ned_m, (3,), "положение эталона")
          - _arr(snapshot.position_ned_m, (3,), "положение оцененное"),
        _arr(truth.velocity_ned_m_s, (3,), "скорость эталона")
          - _arr(snapshot.velocity_ned_m_s, (3,), "скорость оцененная"),
        right_attitude_error_rotation_vector(
            snapshot.quaternion_body_to_ned, truth.quaternion_body_to_ned,
        ),
        _arr(truth_gyro_bias_rad_s, (3,), "эталон гироскопов")
          - _arr(snapshot.gyro_bias_rad_s, (3,), "смещения гироскопов"),
        _arr(truth_accel_bias_m_s2, (3,), "эталон акселерометров")
          - _arr(snapshot.accel_bias_m_s2, (3,), "смещения акселерометров"),
    ))
    return float(error @ np.linalg.solve(cov, error))


def observability_velocity_only(
    samples: Sequence[LinearizationSample], *, rank_rtol: float = 1e-9,
) -> dict:
    """Local linearized observability of initial 15-state error.

    Rows at each time are H_k Phi(t_k,t_0), where H selects only VN/VE.
    This is a local algebraic test for the supplied motion linearization,
    NOT a global/nonlinear observability proof or a flight validation.
    """
    if not samples or not (0 < rank_rtol < 0.1):
        raise ValueError("Нужно непустое окно движения и допустимый порог")
    total = np.eye(15)
    h = np.zeros((2, 15))
    h[:, 3:5] = np.eye(2)
    rows = [h.copy()]
    for item in samples:
        if not isfinite(item.dt_s) or not (0 < item.dt_s <= 0.05):
            raise ValueError("Недопустимый временной шаг наблюдаемости")
        rotation = _arr(item.rotation_body_to_ned, (3, 3), "ориентация")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3):
            raise ValueError("Матрица ориентации не ортогональна")
        if abs(float(np.linalg.det(rotation)) - 1) > 1e-3:
            raise ValueError("Матрица ориентации не является поворотом")
        omega = _arr(item.omega_body_rad_s, (3,), "угловая скорость")
        force = _arr(item.specific_force_body_m_s2, (3,), "удельная сила")
        f = continuous_error_jacobian(rotation, omega, force)
        total = exp_transition(f, item.dt_s) @ total
        rows.append(h @ total)
    stack = np.vstack(rows)
    singular = np.linalg.svd(stack, compute_uv=False)
    scale = float(singular[0]) if len(singular) else 0.0
    rank = int(np.sum(singular > rank_rtol * scale)) if scale > 0 else 0
    nullity = 15 - rank
    # p_N,p_E,p_D have no effect on velocity-only measurements:
    # the absolute translation gauge cannot be determined.
    gauge_columns_zero = bool(np.allclose(stack[:, :3], 0.0, atol=1e-10))
    return {
        "state_dimension": 15,
        "measurement_dimension": 2,
        "samples": len(samples),
        "rank": rank,
        "unobservable_dimension": nullity,
        "translation_gauge_unobservable": gauge_columns_zero,
        "rank_relative_threshold": rank_rtol,
        "singular_values": singular.tolist(),
        "warning": (
            "Локальная линеаризованная наблюдаемость для заданного движения; "
            "зависит от масштабирования состояний, возбуждения и численного порога."
        ),
    }


def flight_level_summary(
    matched: dict[str, Sequence[dict]], *,
    required_flights: int = 3,
    min_samples_per_flight: int = 30,
    bootstrap_repetitions: int = 500,
) -> dict:
    """Aggregate by independent flight, never treat time-correlated frames as IID.

    Bootstraps FLIGHT-level mean NEES, not samples within a trajectory.
    Flags are screening flags, not certification or confidence guarantees.
    """
    if not matched or required_flights < 2 or min_samples_per_flight < 2:
        raise ValueError("Некорректный набор эталонных полетов")
    if not 100 <= bootstrap_repetitions <= 10000:
        raise ValueError("Недопустимое число повторов выборки")
    details = {}
    means = []
    weighted_errors = []
    for flight, rows in sorted(matched.items()):
        if not flight.strip() or not rows:
            raise ValueError("Пустое обозначение или пустые отсчеты полета")
        times = [x["timestamp_us"] for x in rows]
        if any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError("Внутри полета время должно строго возрастать")
        nees = np.asarray([x["nees_vne_2d"] for x in rows], dtype=float)
        err = np.asarray([x["horizontal_speed_error_m_s"] for x in rows], dtype=float)
        if not np.isfinite(nees).all() or not np.isfinite(err).all():
            raise ValueError("Неконечные показатели ошибок")
        means.append(float(np.mean(nees)))
        weighted_errors.extend(float(x*x) for x in err)
        details[flight] = {
            "samples": len(rows),
            "mean_nees_2d": float(np.mean(nees)),
            "fraction_inside_95pct_2d": float(np.mean(nees <= CHI2_2_95)),
            "velocity_rmse_m_s": sqrt(float(np.mean(err**2))),
            "sufficient_within_flight_for_screening": (
                len(rows) >= min_samples_per_flight
            ),
        }
    adequate = (
        len(details) >= required_flights and
        all(x["samples"] >= min_samples_per_flight for x in details.values())
    )
    ci = None
    if len(means) >= required_flights:
        rng = np.random.default_rng(1729)
        means_a = np.asarray(means)
        sampled_means = np.mean(rng.choice(
            means_a, size=(bootstrap_repetitions, len(means_a)), replace=True
        ), axis=1)
        ci = [float(v) for v in np.quantile(sampled_means, [0.025, 0.975])]
    return {
        "independent_flights": len(details),
        "total_matched_samples": sum(x["samples"] for x in details.values()),
        "flights": details,
        "unweighted_mean_of_flight_nees": float(np.mean(means)),
        "flight_bootstrap_95pct_interval_for_mean": ci,
        "pooled_velocity_rmse_m_s": sqrt(float(np.mean(weighted_errors))),
        "sufficient_for_preliminary_screening": adequate,
        "validated": False,
        "warning": (
            "Без реальных независимых траекторий и проверки корреляции "
            "инноваций нельзя утверждать статистическую состоятельность."
        ),
    }
