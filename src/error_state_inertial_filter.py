"""15-мерный инерциальный фильтр ошибок состояния (ESKF), офлайн-прототип.

Состояние ошибки: [δp_NED(3), δv_NED(3), δθ_body(3),
                   δb_gyro_body(3), δb_accel_body(3)].
Номинал: p, v, q(FRD->NED), b_gyro, b_accel.

δθ — ПРАВАЯ ошибка ориентации: R_true = R_nom Exp([δθ]x).
Визуальная коррекция использует только скорость VN,VE центра масс:
при этом полная наблюдаемость 15 состояний НЕ гарантирована.
Географической привязки и обработки запаздывающих измерений нет.
Результат НЕ подключен к PX4 и не является сертифицированной VIO.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

from .horizontal_visual_inertial_filter import VisualVelocityMeasurement
from .imu_preintegration import (
    GRAVITY_NED, ImuBias, ImuReading, _unit_quaternion, _vector,
    quat_exponential, quat_product, rotation_body_to_ned,
)

P = slice(0, 3)
V = slice(3, 6)
TH = slice(6, 9)
BG = slice(9, 12)
BA = slice(12, 15)
_FATAL = frozenset({
    "IMU_INVALID_OR_GAP", "NUMERICAL_FAILURE", "BIAS_LIMIT_EXCEEDED",
    "POSTERIOR_INVALID",
})


def skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = _vector(vector, 3, "вектор кососимметричной матрицы")
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def continuous_error_jacobian(
    rotation_nb: np.ndarray, omega_body: np.ndarray,
    force_body: np.ndarray,
) -> np.ndarray:
    """Линеаризация правоинвариантной ошибки [δp,δv,δθ,δbg,δba]."""
    r = np.asarray(rotation_nb, dtype=float)
    if r.shape != (3, 3) or not np.isfinite(r).all():
        raise ValueError("Некорректная матрица ориентации")
    omega = _vector(omega_body, 3, "угловая скорость")
    force = _vector(force_body, 3, "удельная сила")
    f = np.zeros((15, 15))
    f[P, V] = np.eye(3)
    f[V, TH] = -r @ skew(force)
    f[V, BA] = -r
    f[TH, TH] = -skew(omega)
    f[TH, BG] = -np.eye(3)
    return f


def exp_transition(f: np.ndarray, dt: float) -> np.ndarray:
    """Ряд экспоненты матрицы до 4-го порядка на малом dt <= 0.05 с."""
    if f.shape != (15, 15) or not (isfinite(dt) and 0 <= dt <= 0.05):
        raise ValueError("Некорректные параметры дискретизации")
    a = f * dt
    a2 = a @ a
    a3 = a2 @ a
    a4 = a3 @ a
    return (np.eye(15) + a + 0.5 * a2 +
            a3 / 6.0 + a4 / 24.0)


@dataclass(frozen=True)
class ImuNoiseDensity:
    """Спектральные плотности белых шумов (на sqrt(Hz) в принятой модели)."""
    gyro_rad_s_sqrt_hz: float
    accel_m_s2_sqrt_hz: float
    gyro_bias_walk_rad_s2_sqrt_hz: float
    accel_bias_walk_m_s3_sqrt_hz: float

    def valid(self) -> bool:
        return all(isfinite(v) and 0 < v < 100.0 for v in vars(self).values())


def discrete_process_covariance(
    f: np.ndarray, r_nb: np.ndarray, dt: float,
    noise: ImuNoiseDensity,
) -> np.ndarray:
    """Интеграл Phi(t) G Qc G^T Phi(t)^T, 3-точечная Гаусс-квадратура.

    Положительная полуопределенность сохраняется суммой матриц вида B B^T.
    Знаки шумового воздействия не влияют на диагональные источники,
    предполагаемые независимыми. Временные корреляции здесь не учитываются.
    """
    if not noise.valid() or not (0 < dt <= 0.05):
        raise ValueError("Некорректные плотности шума или шаг ИИМ")
    g = np.zeros((15, 12))
    g[V, 0:3] = -r_nb
    g[TH, 3:6] = -np.eye(3)
    g[BG, 6:9] = np.eye(3)
    g[BA, 9:12] = np.eye(3)
    sigmas = np.repeat([
        noise.accel_m_s2_sqrt_hz,
        noise.gyro_rad_s_sqrt_hz,
        noise.gyro_bias_walk_rad_s2_sqrt_hz,
        noise.accel_bias_walk_m_s3_sqrt_hz,
    ], 3)
    b = g * sigmas[None, :]
    qd = np.zeros((15, 15))
    # Gauss-Legendre nodes on [0,dt].
    nodes = (0.0, -np.sqrt(3. / 5.), np.sqrt(3. / 5.))
    weights = (8. / 9., 5. / 9., 5. / 9.)
    for node, weight in zip(nodes, weights):
        tau = 0.5 * dt * (node + 1.)
        phi = exp_transition(f, tau)
        z = phi @ b
        qd += (0.5 * dt * weight) * (z @ z.T)
    return 0.5 * (qd + qd.T)


@dataclass(frozen=True)
class EskfSnapshot:
    timestamp_us: int
    status: str
    position_ned_m: tuple[float, float, float]
    velocity_ned_m_s: tuple[float, float, float]
    quaternion_body_to_ned: tuple[float, float, float, float]
    gyro_bias_rad_s: tuple[float, float, float]
    accel_bias_m_s2: tuple[float, float, float]
    covariance_15x15: tuple[tuple[float, ...], ...]
    last_nis: float
    accepted_visual_count: int
    rejected_visual_count: int


class InertialErrorStateFilter:
    """15D ESKF с дискретным прогнозом ИИМ и коррекцией скорости VN/VE.

    Необратимые ошибки ИИМ и численные отказы блокируют оценивание
    до создания/явной реинициализации нового фильтра.
    Точки кадра и ИИМ должны быть уже приведены к одной шкале времени;
    обработка внеочередных визуальных измерений пока запрещена.
    """

    def __init__(
        self,
        *,
        quaternion_body_to_ned: tuple[float, float, float, float],
        initial_bias: ImuBias,
        noise: ImuNoiseDensity,
        initial_error_sigma: tuple[float, ...] = (
            10., 10., 10., 1., 1., 1.,
            0.1, 0.1, 0.1, 0.02, 0.02, 0.02,
            0.2, 0.2, 0.2,
        ),
        max_imu_dt_s: float = 0.05,
        nis_gate: float = 9.210340371976184,
        max_gyro_bias_rad_s: float = 0.3,
        max_accel_bias_m_s2: float = 3.0,
    ) -> None:
        if not noise.valid() or not 0.001 <= max_imu_dt_s <= 0.05:
            raise ValueError("Недопустимые характеристики шума/интервала ИИМ")
        if not all(isfinite(v) and v > 0 for v in (
            nis_gate, max_gyro_bias_rad_s, max_accel_bias_m_s2
        )):
            raise ValueError("Недопустимый порог инновации/смещений нуля")
        sigmas = _vector(initial_error_sigma, 15, "начальные сигмы состояния")
        if np.min(sigmas) <= 0:
            raise ValueError("Начальная ковариация должна быть положительной")
        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.q = _unit_quaternion(quaternion_body_to_ned)
        self.bg = _vector(initial_bias.gyro_body_rad_s, 3, "смещение гироскопов")
        self.ba = _vector(initial_bias.accel_body_m_s2, 3, "смещение акселерометров")
        self.covariance = np.diag(sigmas ** 2)
        self.noise = noise
        self.max_imu_dt_s = max_imu_dt_s
        self.nis_gate = nis_gate
        self.max_bg = max_gyro_bias_rad_s
        self.max_ba = max_accel_bias_m_s2
        self._last: ImuReading | None = None
        self._last_visual_us: int | None = None
        self._accepted = 0
        self._rejected = 0
        self._nis = float("nan")
        self._status = "WAITING_FOR_IMU"

    def snapshot(self) -> EskfSnapshot:
        return EskfSnapshot(
            self._last.timestamp_us if self._last is not None else 0,
            self._status, tuple(self.p), tuple(self.v), tuple(self.q),
            tuple(self.bg), tuple(self.ba),
            tuple(tuple(float(y) for y in row) for row in self.covariance),
            self._nis, self._accepted, self._rejected,
        )

    def _fail(self, reason: str) -> EskfSnapshot:
        self._status = reason
        return self.snapshot()

    def predict(self, reading: ImuReading) -> EskfSnapshot:
        if self._status in _FATAL:
            return self.snapshot()
        try:
            if not isinstance(reading.timestamp_us, int) or reading.timestamp_us <= 0:
                raise ValueError("Время ИИМ")
            omega_raw = _vector(reading.gyro_body_rad_s, 3, "гироскоп")
            force_raw = _vector(reading.specific_force_body_m_s2, 3, "акселерометр")
            if self._last is None:
                self._last = reading
                self._status = "INERTIAL_ONLY"
                return self.snapshot()
            dt = (reading.timestamp_us - self._last.timestamp_us) * 1e-6
            if not 0 < dt <= self.max_imu_dt_s:
                raise ValueError("Непрерывность ИИМ")
            omega = (omega_raw + _vector(
                self._last.gyro_body_rad_s, 3, "предыдущий гироскоп"
            )) * 0.5 - self.bg
            f_body = (force_raw + _vector(
                self._last.specific_force_body_m_s2, 3, "предыдущий акселерометр"
            )) * 0.5 - self.ba
            qmid = _unit_quaternion(quat_product(
                self.q, quat_exponential(0.5 * dt * omega)
            ))
            rmid = rotation_body_to_ned(qmid)
            a_nav = rmid @ f_body + GRAVITY_NED
            # Jacobian at midpoint before replacing the nominal orientation.
            a = continuous_error_jacobian(rmid, omega, f_body)
            phi = exp_transition(a, dt)
            qd = discrete_process_covariance(a, rmid, dt, self.noise)
            p_new = self.p + dt * self.v + 0.5 * dt * dt * a_nav
            v_new = self.v + dt * a_nav
            q_new = _unit_quaternion(quat_product(
                self.q, quat_exponential(dt * omega)
            ))
            cov_new = phi @ self.covariance @ phi.T + qd
            cov_new = 0.5 * (cov_new + cov_new.T)
            if (not np.isfinite(cov_new).all() or
                    np.linalg.eigvalsh(cov_new)[0] <= 0 or
                    not np.isfinite(p_new).all() or not np.isfinite(v_new).all()):
                return self._fail("NUMERICAL_FAILURE")
            self.p, self.v, self.q = p_new, v_new, q_new
            self.covariance = cov_new
            self._last = reading
            self._status = "INERTIAL_ONLY"
        except (ValueError, np.linalg.LinAlgError):
            return self._fail("IMU_INVALID_OR_GAP")
        return self.snapshot()

    def update_visual_velocity(
        self, measurement: VisualVelocityMeasurement,
    ) -> EskfSnapshot:
        if self._status in _FATAL:
            return self.snapshot()
        if self._last is None:
            return self._fail("NO_IMU_REFERENCE")
        stamp = measurement.timestamp_us
        if not isinstance(stamp, int) or stamp != self._last.timestamp_us:
            return self._fail("UNALIGNED_VISUAL_TIMESTAMP")
        if self._last_visual_us is not None and stamp <= self._last_visual_us:
            return self._fail("REPEATED_VISUAL_TIMESTAMP")
        try:
            z = _vector((measurement.north_m_s, measurement.east_m_s), 2,
                        "визуальная скорость")
            r = np.asarray(measurement.covariance_ne_m2_s2, dtype=float)
            if (r.shape != (2, 2) or not np.isfinite(r).all() or
                    not np.allclose(r, r.T, atol=1e-10) or
                    np.linalg.eigvalsh(r)[0] <= 0):
                raise ValueError("Матрица ошибок визуального измерения")
            h = np.zeros((2, 15))
            h[:, 3:5] = np.eye(2)
            innovation = z - self.v[:2]
            s = h @ self.covariance @ h.T + r
            l = np.linalg.cholesky(s)
            normalized = np.linalg.solve(l, innovation)
            nis = float(normalized @ normalized)
            self._nis = nis
            self._last_visual_us = stamp
            if nis > self.nis_gate:
                self._rejected += 1
                return self._fail("VISUAL_OUTLIER_REJECTED")
            k = np.linalg.solve(s, h @ self.covariance).T
            dx = k @ innovation
            candidate_bg = self.bg + dx[BG]
            candidate_ba = self.ba + dx[BA]
            if (np.linalg.norm(candidate_bg) > self.max_bg or
                    np.linalg.norm(candidate_ba) > self.max_ba):
                return self._fail("BIAS_LIMIT_EXCEEDED")

            a = np.eye(15) - k @ h
            cov = a @ self.covariance @ a.T + k @ r @ k.T  # Joseph update
            # Reset Jacobian for right-multiplicative orientation error.
            reset = np.eye(15)
            reset[TH, TH] -= 0.5 * skew(dx[TH])
            cov = reset @ cov @ reset.T
            cov = 0.5 * (cov + cov.T)
            q_new = _unit_quaternion(quat_product(
                self.q, quat_exponential(dx[TH])
            ))
            if not np.isfinite(cov).all() or np.linalg.eigvalsh(cov)[0] <= 0:
                return self._fail("POSTERIOR_INVALID")
            self.p += dx[P]
            self.v += dx[V]
            self.q = q_new
            self.bg = candidate_bg
            self.ba = candidate_ba
            self.covariance = cov
            self._accepted += 1
            return self._fail("VISUAL_CORRECTED")
        except (ValueError, np.linalg.LinAlgError):
            self._rejected += 1
            return self._fail("VISUAL_COVARIANCE_INVALID")
