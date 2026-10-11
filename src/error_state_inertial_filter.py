"""15-мерный инерциальный фильтр ошибок состояния (ESKF), офлайн-прототип.

Состояние ошибки: [δp_NED(3), δv_NED(3), δθ_body(3),
                   δb_gyro_body(3), δb_accel_body(3)].
Номинал: p, v, q(FRD->NED), b_gyro, b_accel.

δθ — ПРАВАЯ ошибка ориентации: R_true = R_nom Exp([δθ]x).
Коррекции (те же модели, что в firmware/rtos_core/src/eskf15.cpp):

- VN, VE центра масс (``update_visual_velocity``);
- скорость в осях корпуса FRD x, y (``update_body_velocity``):
  h = [R^T v]_xy, якобиан по δθ = [R^T v]x — связывает камеру
  и ИИМ без внешней ориентации, курс наблюдаем при маневрах;
- курс (``update_yaw``), например по ортофотоплану;
- положение N, E по ортофотоплану (``update_map_position``) с
  ковариационным пересечением при неизвестной корреляции с VO.

Полная наблюдаемость 15 состояний НЕ гарантирована (см.
``src/eskf_observability.py``). Результат НЕ передается в EKF2.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, isfinite, sin, sqrt
from typing import Sequence

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
GRAVITY = 9.80665
EARTH_RATE_RAD_S = 7.2921159e-5
CHI2_1_99 = 6.6348966010212145
# Kind order == processing order for equal timestamps (same as C++).
KIND_ORDER = {"velocity_ne": 0, "body_velocity_xy": 1, "yaw": 2, "position_ne": 3,
              "ground_point_ne": 4}
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
class BodyVelocityMeasurement:
    """Скорость центра масс в осях корпуса FRD x, y (камера + дальномер)."""
    timestamp_us: int
    x_m_s: float
    y_m_s: float
    covariance_xy_m2_s2: tuple[tuple[float, float], tuple[float, float]]


@dataclass(frozen=True)
class YawMeasurement:
    """Курс оси x FRD от севера, рад (ортофотоплан, магнитометр)."""
    timestamp_us: int
    yaw_rad: float
    variance_rad2: float


@dataclass(frozen=True)
class MapPositionMeasurement:
    """Положение N, E по ортофотоплану в локальной СК фильтра.

    ci_weight: 1 — независимое измерение (стандартная коррекция);
    (0, 1) — ковариационное пересечение по всему состоянию: P/w, R/(1-w);
    <= 0 — w выбирается на сетке 0,05..0,95 минимизацией det полной
    апостериорной ковариации (``choose_ci_weight``).
    """
    timestamp_us: int
    north_m: float
    east_m: float
    covariance_ne_m2: tuple[tuple[float, float], tuple[float, float]]
    ci_weight: float = 1.0


@dataclass(frozen=True)
class MapGroundPointMeasurement:
    """N, E точки земли на оптической оси камеры (привязка главной точки).

    Не зависит от оценки ориентации. Модель:
    z = p_NE + h (R d)_NE / (R d)_D, d — ось камеры в осях FRD,
    h — высота над плоскостью по дальномеру. Связывает карту с наклоном.
    """
    timestamp_us: int
    north_m: float
    east_m: float
    covariance_ne_m2: tuple[tuple[float, float], tuple[float, float]]
    height_agl_m: float
    optical_axis_body: tuple[float, float, float] = (0., 0., 1.)
    ci_weight: float = 1.0


def measurement_kind(measurement: object) -> str:
    if isinstance(measurement, VisualVelocityMeasurement):
        return "velocity_ne"
    if isinstance(measurement, BodyVelocityMeasurement):
        return "body_velocity_xy"
    if isinstance(measurement, YawMeasurement):
        return "yaw"
    if isinstance(measurement, MapPositionMeasurement):
        return "position_ne"
    if isinstance(measurement, MapGroundPointMeasurement):
        return "ground_point_ne"
    raise TypeError("Неизвестный тип измерения")


def choose_ci_weight(hph: np.ndarray, measurement: np.ndarray, *,
                     state_dimension: int = 15) -> float:
    """Вес ковариационного пересечения по полному состоянию.

    Минимизируется ln det P+(w) = -n ln w + ln det(R/(1-w))
    - ln det(HPH^T/w + R/(1-w)) на сетке w = 0.05..0.95 (как в C++).
    Критерий только по блоку положения выбирал бы w -> 0 и раздувал
    все состояния в 1/w раз при каждой привязке к карте.
    """
    best_w, best = 0.5, np.inf
    for step in range(5, 96):
        w = step / 100.
        a = 1. / (1. - w)
        rd = float(np.linalg.det(measurement)) * a * a
        sd = float(np.linalg.det(hph / w + measurement * a))
        if rd <= 0 or sd <= 0:
            continue
        cost = -state_dimension * np.log(w) + np.log(rd) - np.log(sd)
        if cost < best:
            best, best_w = cost, w
    return best_w


@dataclass(frozen=True)
class AlignmentResult:
    status: str
    quaternion_body_to_ned: tuple[float, float, float, float] | None = None
    gyro_bias_rad_s: tuple[float, float, float] | None = None
    initial_error_sigma: tuple[float, ...] | None = None


def static_alignment(
    readings: Sequence[ImuReading], *, yaw_rad: float, yaw_sigma_rad: float,
    prior_sigma: tuple[float, ...] = (
        10., 10., 10., 1., 1., 1., 0.1, 0.1, 0.1,
        0.02, 0.02, 0.02, 0.2, 0.2, 0.2,
    ),
    min_samples: int = 50, max_gyro_std_rad_s: float = 0.01,
    max_accel_std_m_s2: float = 0.15, max_gravity_error_m_s2: float = 0.5,
    max_gyro_bias_rad_s: float = 0.3,
) -> AlignmentResult:
    """Грубая выставка на неподвижном основании (как staticAlignment в C++).

    Крен/тангаж — по средней удельной силе, смещение гироскопов — по
    среднему. Курс по неподвижной ИИМ MEMS ненаблюдаем и задается
    извне вместе с СКО. Смещение акселерометров неотделимо от наклона:
    его априорная СКО добавляется к СКО крена/тангажа.
    """
    if not (isfinite(yaw_rad) and isfinite(yaw_sigma_rad) and yaw_sigma_rad > 0
            and min_samples >= 2):
        return AlignmentResult("INVALID_INPUT")
    if len(readings) < min_samples:
        return AlignmentResult("TOO_FEW_SAMPLES")
    try:
        w = np.array([_vector(r.gyro_body_rad_s, 3, "гироскоп") for r in readings])
        f = np.array([_vector(r.specific_force_body_m_s2, 3, "акселерометр")
                      for r in readings])
    except ValueError:
        return AlignmentResult("INVALID_INPUT")
    stamps = [r.timestamp_us for r in readings]
    if any(b <= a for a, b in zip(stamps, stamps[1:])) or stamps[0] <= 0:
        return AlignmentResult("INVALID_INPUT")
    n = len(readings)
    mw, mf = w.mean(axis=0), f.mean(axis=0)
    sw = sqrt(float(np.sum((w - mw) ** 2)) / (3. * (n - 1)))
    sf = sqrt(float(np.sum((f - mf) ** 2)) / (3. * (n - 1)))
    if sw > max_gyro_std_rad_s or sf > max_accel_std_m_s2 or (
            float(np.linalg.norm(mw)) > max_gyro_bias_rad_s):
        return AlignmentResult("NOT_STATIONARY")
    if abs(float(np.linalg.norm(mf)) - GRAVITY) > max_gravity_error_m_s2:
        return AlignmentResult("GRAVITY_MISMATCH")
    roll = atan2(-mf[1], -mf[2])
    pitch = atan2(mf[0], sqrt(mf[1] ** 2 + mf[2] ** 2))
    cr, sr = cos(.5 * roll), sin(.5 * roll)
    cp, sp = cos(.5 * pitch), sin(.5 * pitch)
    cy, sy = cos(.5 * yaw_rad), sin(.5 * yaw_rad)
    q = (cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
         cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy)
    sigma = list(_vector(prior_sigma, 15, "априорные СКО"))
    tilt = sqrt(sf * sf / n / GRAVITY ** 2 + (sigma[12] / GRAVITY) ** 2)
    sigma[6] = sigma[7] = max(tilt, 1e-4)
    sigma[8] = yaw_sigma_rad
    gb = sqrt(sw * sw / n + EARTH_RATE_RAD_S ** 2)
    for k in range(9, 12):
        sigma[k] = max(gb, 1e-6)
    for k in range(3, 6):
        sigma[k] = min(sigma[k], 0.05)
    return AlignmentResult("ALIGNED", q, tuple(float(x) for x in mw), tuple(sigma))


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
        nis_gate_1d: float = CHI2_1_99,
        max_gyro_bias_rad_s: float = 0.3,
        max_accel_bias_m_s2: float = 3.0,
        initial_position_ned_m: tuple[float, float, float] = (0., 0., 0.),
    ) -> None:
        if not noise.valid() or not 0.001 <= max_imu_dt_s <= 0.05:
            raise ValueError("Недопустимые характеристики шума/интервала ИИМ")
        if not all(isfinite(v) and v > 0 for v in (
            nis_gate, nis_gate_1d, max_gyro_bias_rad_s, max_accel_bias_m_s2
        )):
            raise ValueError("Недопустимый порог инновации/смещений нуля")
        sigmas = _vector(initial_error_sigma, 15, "начальные сигмы состояния")
        if np.min(sigmas) <= 0:
            raise ValueError("Начальная ковариация должна быть положительной")
        self.p = _vector(initial_position_ned_m, 3, "начальное положение")
        self.v = np.zeros(3)
        self.q = _unit_quaternion(quaternion_body_to_ned)
        self.bg = _vector(initial_bias.gyro_body_rad_s, 3, "смещение гироскопов")
        self.ba = _vector(initial_bias.accel_body_m_s2, 3, "смещение акселерометров")
        self.covariance = np.diag(sigmas ** 2)
        self.noise = noise
        self.max_imu_dt_s = max_imu_dt_s
        self.nis_gate = nis_gate
        self.nis_gate_1d = nis_gate_1d
        self.max_bg = max_gyro_bias_rad_s
        self.max_ba = max_accel_bias_m_s2
        self._last: ImuReading | None = None
        self._last_measurement_us: dict[str, int] = {}
        self.last_ci_weight = 1.0
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

    def apply(self, measurement: object) -> EskfSnapshot:
        """Коррекция по измерению любого поддерживаемого типа."""
        kind = measurement_kind(measurement)
        if kind == "velocity_ne":
            return self.update_visual_velocity(measurement)
        if kind == "body_velocity_xy":
            return self.update_body_velocity(measurement)
        if kind == "yaw":
            return self.update_yaw(measurement)
        if kind == "ground_point_ne":
            return self.update_map_ground_point(measurement)
        return self.update_map_position(measurement)

    def _admit(self, kind: str, stamp: object) -> str | None:
        if self._status in _FATAL:
            return "FATAL"
        if self._last is None:
            return "NO_IMU_REFERENCE"
        if not isinstance(stamp, int) or stamp != self._last.timestamp_us:
            return "UNALIGNED_VISUAL_TIMESTAMP"
        last = self._last_measurement_us.get(kind)
        if last is not None and stamp <= last:
            return "REPEATED_VISUAL_TIMESTAMP"
        return None

    def _correct(
        self, kind: str, stamp: int, innovation: np.ndarray, h: np.ndarray,
        r: np.ndarray, *, gate: float, corrected: str, outlier: str,
        ci_weight: float = 1.0,
    ) -> EskfSnapshot:
        """Общая коррекция: NIS по честной S = HPH^T + R, форма Джозефа.

        При ci_weight < 1: (I-KH)(P/w)(I-KH)^T + K(R/(1-w))K^T,
        K = (P/w)H^T S_ci^-1 — ковариационное пересечение по полному
        состоянию для оценок с неизвестной взаимной корреляцией.
        """
        s = h @ self.covariance @ h.T + r
        l = np.linalg.cholesky(s)
        normalized = np.linalg.solve(l, innovation)
        nis = float(normalized @ normalized)
        self._nis = nis
        self._last_measurement_us[kind] = stamp
        if nis > gate:
            self._rejected += 1
            return self._fail(outlier)
        w = 1.0
        if kind in ("position_ne", "ground_point_ne") and ci_weight < 1.0:
            w = ci_weight if ci_weight > 0 else choose_ci_weight(s - r, r)
        pw = self.covariance / w
        rw = r / (1.0 - w) if w < 1.0 else r
        s_ci = h @ pw @ h.T + rw
        k = np.linalg.solve(s_ci, h @ pw).T
        dx = k @ innovation
        candidate_bg = self.bg + dx[BG]
        candidate_ba = self.ba + dx[BA]
        if (np.linalg.norm(candidate_bg) > self.max_bg or
                np.linalg.norm(candidate_ba) > self.max_ba):
            return self._fail("BIAS_LIMIT_EXCEEDED")
        a = np.eye(15) - k @ h
        cov = a @ pw @ a.T + k @ rw @ k.T  # Joseph update
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
        self.last_ci_weight = w
        self._accepted += 1
        return self._fail(corrected)

    @staticmethod
    def _covariance2(value: object) -> np.ndarray:
        r = np.asarray(value, dtype=float)
        if (r.shape != (2, 2) or not np.isfinite(r).all() or
                not np.allclose(r, r.T, atol=1e-10) or
                np.linalg.eigvalsh(r)[0] <= 0):
            raise ValueError("Матрица ошибок измерения")
        return r

    def update_visual_velocity(
        self, measurement: VisualVelocityMeasurement,
    ) -> EskfSnapshot:
        refused = self._admit("velocity_ne", measurement.timestamp_us)
        if refused == "FATAL":
            return self.snapshot()
        if refused is not None:
            return self._fail(refused)
        try:
            z = _vector((measurement.north_m_s, measurement.east_m_s), 2,
                        "визуальная скорость")
            r = self._covariance2(measurement.covariance_ne_m2_s2)
            h = np.zeros((2, 15))
            h[:, 3:5] = np.eye(2)
            return self._correct(
                "velocity_ne", measurement.timestamp_us, z - self.v[:2], h, r,
                gate=self.nis_gate, corrected="VISUAL_CORRECTED",
                outlier="VISUAL_OUTLIER_REJECTED")
        except (ValueError, np.linalg.LinAlgError):
            self._rejected += 1
            return self._fail("VISUAL_COVARIANCE_INVALID")

    def update_body_velocity(
        self, measurement: BodyVelocityMeasurement,
    ) -> EskfSnapshot:
        refused = self._admit("body_velocity_xy", measurement.timestamp_us)
        if refused == "FATAL":
            return self.snapshot()
        if refused is not None:
            return self._fail(refused)
        try:
            z = _vector((measurement.x_m_s, measurement.y_m_s), 2,
                        "скорость в осях корпуса")
            r = self._covariance2(measurement.covariance_xy_m2_s2)
            rot = rotation_body_to_ned(self.q)
            u = rot.T @ self.v
            h = np.zeros((2, 15))
            h[:, V] = rot.T[0:2, :]
            h[:, TH] = skew(u)[0:2, :]
            return self._correct(
                "body_velocity_xy", measurement.timestamp_us, z - u[0:2], h, r,
                gate=self.nis_gate, corrected="VISUAL_CORRECTED",
                outlier="VISUAL_OUTLIER_REJECTED")
        except (ValueError, np.linalg.LinAlgError):
            self._rejected += 1
            return self._fail("VISUAL_COVARIANCE_INVALID")

    def update_yaw(self, measurement: YawMeasurement) -> EskfSnapshot:
        refused = self._admit("yaw", measurement.timestamp_us)
        if refused == "FATAL":
            return self.snapshot()
        if refused is not None:
            return self._fail(refused)
        try:
            yaw = float(measurement.yaw_rad)
            var = float(measurement.variance_rad2)
            if not (isfinite(yaw) and isfinite(var) and 1e-12 < var < 10.):
                raise ValueError("Дисперсия курса")
            rot = rotation_body_to_ned(self.q)
            den = rot[0, 0] ** 2 + rot[1, 0] ** 2
            if den <= 1e-6:
                raise ValueError("Курс не определен при вертикальной оси x")
            residual = atan2(sin(yaw - atan2(rot[1, 0], rot[0, 0])),
                             cos(yaw - atan2(rot[1, 0], rot[0, 0])))
            h = np.zeros((1, 15))
            h[0, 7] = (rot[1, 0] * rot[0, 2] - rot[0, 0] * rot[1, 2]) / den
            h[0, 8] = (rot[0, 0] * rot[1, 1] - rot[1, 0] * rot[0, 1]) / den
            return self._correct(
                "yaw", measurement.timestamp_us, np.array([residual]), h,
                np.array([[var]]), gate=self.nis_gate_1d,
                corrected="YAW_CORRECTED", outlier="YAW_OUTLIER_REJECTED")
        except (ValueError, np.linalg.LinAlgError):
            self._rejected += 1
            return self._fail("VISUAL_COVARIANCE_INVALID")

    def update_map_position(
        self, measurement: MapPositionMeasurement,
    ) -> EskfSnapshot:
        refused = self._admit("position_ne", measurement.timestamp_us)
        if refused == "FATAL":
            return self.snapshot()
        if refused is not None:
            return self._fail(refused)
        try:
            w = float(measurement.ci_weight)
            if not isfinite(w) or w > 1. or (0. < w < 1. and not .01 <= w <= .99):
                raise ValueError("Вес ковариационного пересечения")
            z = _vector((measurement.north_m, measurement.east_m), 2,
                        "положение по карте")
            r = self._covariance2(measurement.covariance_ne_m2)
            h = np.zeros((2, 15))
            h[:, 0:2] = np.eye(2)
            return self._correct(
                "position_ne", measurement.timestamp_us, z - self.p[:2], h, r,
                gate=self.nis_gate, corrected="POSITION_CORRECTED",
                outlier="POSITION_OUTLIER_REJECTED", ci_weight=w)
        except (ValueError, np.linalg.LinAlgError):
            self._rejected += 1
            return self._fail("VISUAL_COVARIANCE_INVALID")

    def update_map_ground_point(
        self, measurement: MapGroundPointMeasurement,
    ) -> EskfSnapshot:
        refused = self._admit("ground_point_ne", measurement.timestamp_us)
        if refused == "FATAL":
            return self.snapshot()
        if refused is not None:
            return self._fail(refused)
        try:
            w = float(measurement.ci_weight)
            if not isfinite(w) or w > 1. or (0. < w < 1. and not .01 <= w <= .99):
                raise ValueError("Вес ковариационного пересечения")
            height = float(measurement.height_agl_m)
            axis = _vector(measurement.optical_axis_body, 3, "ось камеры")
            norm2 = float(axis @ axis)
            if not (isfinite(height) and 0 < height <= 500. and .81 <= norm2 <= 1.21):
                raise ValueError("Высота или ось камеры")
            db = axis / sqrt(norm2)
            z = _vector((measurement.north_m, measurement.east_m), 2,
                        "точка земли по карте")
            r = self._covariance2(measurement.covariance_ne_m2)
            rot = rotation_body_to_ned(self.q)
            dn = rot @ db
            if dn[2] <= .5:
                raise ValueError("Ось камеры не направлена вниз")
            jac = -rot @ skew(db)
            predicted = self.p[:2] + height * dn[:2] / dn[2]
            h = np.zeros((2, 15))
            h[:, 0:2] = np.eye(2)
            h[:, TH] = height * (jac[:2, :] / dn[2] -
                                 np.outer(dn[:2], jac[2, :]) / dn[2] ** 2)
            return self._correct(
                "ground_point_ne", measurement.timestamp_us, z - predicted, h, r,
                gate=self.nis_gate, corrected="POSITION_CORRECTED",
                outlier="POSITION_OUTLIER_REJECTED", ci_weight=w)
        except (ValueError, np.linalg.LinAlgError):
            self._rejected += 1
            return self._fail("VISUAL_COVARIANCE_INVALID")
