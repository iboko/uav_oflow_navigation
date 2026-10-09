"""Численное интегрирование сырых измерений ИИМ в СК NED.

ИИМ сообщает удельную силу и угловую скорость в связанной СК FRD.
Кватернион [w,x,y,z] переводит FRD -> NED. Гравитация NED [0,0,+g].
Ориентация и смещения нуля НЕ наблюдаемы по ИИМ в одиночку; применяйте
внешнюю инициализацию ориентации и независимую проверку неподвижности.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin, sqrt, isfinite
from typing import Iterable

import numpy as np

GRAVITY_NED = np.array([0.0, 0.0, 9.80665], dtype=np.float64)


@dataclass(frozen=True)
class ImuReading:
    timestamp_us: int
    gyro_body_rad_s: tuple[float, float, float]
    specific_force_body_m_s2: tuple[float, float, float]


@dataclass(frozen=True)
class ImuBias:
    gyro_body_rad_s: tuple[float, float, float] = (0.0, 0.0, 0.0)
    accel_body_m_s2: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class PreintegrationOutput:
    timestamp_us: int
    dt_s: float
    delta_position_ned_m: tuple[float, float, float]
    delta_velocity_ned_m_s: tuple[float, float, float]
    quaternion_body_to_ned: tuple[float, float, float, float]


def _vector(values: object, length: int, label: str) -> np.ndarray:
    try:
        arr = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Неверный вектор: {label}") from exc
    if arr.shape != (length,) or not np.isfinite(arr).all():
        raise ValueError(f"Неконечный или неверного размера вектор: {label}")
    return arr


def _unit_quaternion(q: object) -> np.ndarray:
    arr = _vector(q, 4, "кватернион")
    norm = float(np.linalg.norm(arr))
    if norm < 0.8 or norm > 1.2:
        raise ValueError("Кватернион должен иметь единичную норму")
    return arr / norm


def quat_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w, x, y, z = a
    p, q, r, s = b
    return np.array([
        w * p - x * q - y * r - z * s,
        w * q + x * p + y * s - z * r,
        w * r - x * s + y * p + z * q,
        w * s + x * r - y * q + z * p,
    ], dtype=np.float64)


def quat_exponential(angular_increment_rad: np.ndarray) -> np.ndarray:
    """Кватернион конечного поворота exp(omega*dt/2), без малого угла в нуле."""
    rotation = _vector(angular_increment_rad, 3, "угловое приращение")
    angle = float(np.linalg.norm(rotation))
    if angle < 1e-8:
        factor = 0.5 - angle * angle / 48.0
    else:
        factor = sin(angle / 2.0) / angle
    return np.array([cos(angle / 2.0), *(rotation * factor)], dtype=np.float64)


def rotation_body_to_ned(quaternion: object) -> np.ndarray:
    w, x, y, z = _unit_quaternion(quaternion)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def calibrate_stationary_bias(
    samples: Iterable[ImuReading],
    *,
    known_attitude_body_to_ned: tuple[float, float, float, float],
    confirmed_stationary: bool,
    min_samples: int = 30,
    max_gyro_std_rad_s: float = 0.02,
    max_accel_std_m_s2: float = 0.15,
) -> ImuBias:
    """Оценка смещений нуля ТОЛЬКО на независимо установленной стоянке.

    Требует ориентации по независимому источнику. Одна лишь малая
    дисперсия ИИМ не доказывает неподвижность и не определяет ориентацию.
    """
    if not confirmed_stationary:
        raise ValueError("Неподвижность должна быть подтверждена независимо")
    records = list(samples)
    if len(records) < min_samples or min_samples < 10:
        raise ValueError("Недостаточное количество отсчетов ИИМ")
    t = [r.timestamp_us for r in records]
    if any(not isinstance(v, int) or v <= 0 for v in t) or any(
        b <= a for a, b in zip(t, t[1:])
    ):
        raise ValueError("Неверные временные метки ИИМ")
    gyro = np.array([_vector(r.gyro_body_rad_s, 3, "гироскоп") for r in records])
    accel = np.array([_vector(r.specific_force_body_m_s2, 3, "акселерометр")
                      for r in records])
    if (np.max(np.std(gyro, axis=0)) > max_gyro_std_rad_s or
            np.max(np.std(accel, axis=0)) > max_accel_std_m_s2):
        raise ValueError("Данные ИИМ несовместимы с допущением о неподвижности")
    r_nb = rotation_body_to_ned(known_attitude_body_to_ned)
    expected_stationary_force = r_nb.T @ -GRAVITY_NED
    gyro_bias = gyro.mean(axis=0)
    accel_bias = accel.mean(axis=0) - expected_stationary_force
    return ImuBias(tuple(gyro_bias), tuple(accel_bias))


class ImuPreintegrator:
    """Среднеточечное интегрирование положения, скорости и ориентации.

    Интервал больше max_dt_s переводит объект в состояние отказа до
    явного reset. Не подменяет неопределенный пропуск времени нулевым dt.
    Не содержит ковариации или задержанных измерений.
    """

    def __init__(
        self,
        *,
        quaternion_body_to_ned: tuple[float, float, float, float],
        bias: ImuBias = ImuBias(),
        max_dt_s: float = 0.05,
    ) -> None:
        if not (0.001 <= max_dt_s <= 0.5):
            raise ValueError("Недопустимый максимальный интервал ИИМ")
        self.max_dt_s = max_dt_s
        self.reset(quaternion_body_to_ned=quaternion_body_to_ned, bias=bias)

    def reset(
        self,
        *,
        quaternion_body_to_ned: tuple[float, float, float, float],
        bias: ImuBias = ImuBias(),
    ) -> None:
        self.q = _unit_quaternion(quaternion_body_to_ned)
        self.bg = _vector(bias.gyro_body_rad_s, 3, "смещение гироскопов")
        self.ba = _vector(bias.accel_body_m_s2, 3, "смещение акселерометров")
        self.pos = np.zeros(3, dtype=np.float64)
        self.vel = np.zeros(3, dtype=np.float64)
        self._last: ImuReading | None = None
        self.healthy = True

    def step(self, reading: ImuReading) -> PreintegrationOutput:
        if not self.healthy:
            raise ValueError("Интегрирование остановлено: требуется явный reset")
        if not isinstance(reading.timestamp_us, int) or reading.timestamp_us <= 0:
            self.healthy = False
            raise ValueError("Некорректное время ИИМ")
        _vector(reading.gyro_body_rad_s, 3, "гироскоп")
        _vector(reading.specific_force_body_m_s2, 3, "акселерометр")
        if self._last is None:
            self._last = reading
            return self._output(reading.timestamp_us, 0.0)
        dt = (reading.timestamp_us - self._last.timestamp_us) * 1e-6
        if dt <= 0 or dt > self.max_dt_s:
            self.healthy = False
            raise ValueError("Повтор времени или потеря непрерывности ИИМ")
        omega = 0.5 * (
            _vector(self._last.gyro_body_rad_s, 3, "гироскоп") +
            _vector(reading.gyro_body_rad_s, 3, "гироскоп")
        ) - self.bg
        f = 0.5 * (
            _vector(self._last.specific_force_body_m_s2, 3, "акселерометр") +
            _vector(reading.specific_force_body_m_s2, 3, "акселерометр")
        ) - self.ba
        dq_half = quat_exponential(omega * (dt / 2.0))
        r_mid = rotation_body_to_ned(quat_product(self.q, dq_half))
        a_nav = r_mid @ f + GRAVITY_NED
        old_velocity = self.vel.copy()
        delta_v = a_nav * dt
        delta_p = old_velocity * dt + 0.5 * a_nav * dt * dt
        self.vel += delta_v
        self.pos += delta_p
        self.q = _unit_quaternion(quat_product(self.q, quat_exponential(omega * dt)))
        self._last = reading
        return self._output(reading.timestamp_us, dt, delta_p, delta_v)

    def _output(
        self, timestamp_us: int, dt_s: float,
        delta_p: np.ndarray | None = None,
        delta_v: np.ndarray | None = None,
    ) -> PreintegrationOutput:
        zero = np.zeros(3) if delta_p is None else delta_p
        dv = np.zeros(3) if delta_v is None else delta_v
        return PreintegrationOutput(
            timestamp_us, dt_s, tuple(float(x) for x in zero),
            tuple(float(x) for x in dv),
            tuple(float(x) for x in self.q),
        )
