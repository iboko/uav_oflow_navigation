"""Исследовательский горизонтальный визуально-инерциальный РФК.

Состояние x=[N,E,VN,VE,ba_x,ba_y], где ba_x/ba_y — дополнительные
смещения нуля акселерометров в связанной СК БВС FRD после начальной
калибровки. Ориентация автономно интегрируется по ГИРОСКОПАМ ИИМ и
начальному кватерниону. Видеоканал измеряет только VN/VE в NED.

НЕ является полноценной VIO: нет оценивания ошибок ориентации,
адаптации смещений гироскопов, полноценного 15-мерного ESKF, истории
запаздывающих измерений, карты и независимой абсолютной координаты.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
import numpy as np

from .imu_preintegration import (
    GRAVITY_NED, ImuBias, ImuReading, ImuPreintegrator,
    rotation_body_to_ned, _vector,
)


@dataclass(frozen=True)
class VisualVelocityMeasurement:
    timestamp_us: int
    north_m_s: float
    east_m_s: float
    covariance_ne_m2_s2: tuple[tuple[float, float], tuple[float, float], ...]


@dataclass(frozen=True)
class FusionOutput:
    timestamp_us: int
    status: str
    north_m: float
    east_m: float
    vn_m_s: float
    ve_m_s: float
    accel_bias_x_m_s2: float
    accel_bias_y_m_s2: float
    nis: float
    visual_updates: int
    # This covariance neglects attitude uncertainty. NOT ready for flight control.
    covariance_6x6: tuple[tuple[float, ...], ...]


class HorizontalVisualInertialFilter:
    """Predict at every raw IMU sample; update only at aligned camera timestamps.

    Out-of-sequence camera measurements are rejected; never silently
    corrected at the wrong state time. The startup attitude and static gyro
    bias must be independently supplied. This is an offline prototype.
    """

    def __init__(
        self,
        *,
        quaternion_body_to_ned: tuple[float, float, float, float],
        initial_bias: ImuBias,
        gyro_noise_rad_s: float,
        accel_noise_m_s2: float,
        accel_bias_walk_m_s2_sqrt_s: float,
        max_imu_dt_s: float = 0.05,
        gate_nis: float = 9.21,
        max_estimated_accel_bias_m_s2: float = 1.0,
    ) -> None:
        if not all(
            isfinite(v) and v > 0 for v in (
                gyro_noise_rad_s, accel_noise_m_s2,
                accel_bias_walk_m_s2_sqrt_s,
                gate_nis, max_estimated_accel_bias_m_s2
            )
        ):
            raise ValueError("Дисперсии и границы РФК должны быть положительны")
        self._imu = ImuPreintegrator(
            quaternion_body_to_ned=quaternion_body_to_ned,
            bias=initial_bias, max_dt_s=max_imu_dt_s,
        )
        self._gyro_noise = gyro_noise_rad_s
        self._accel_noise = accel_noise_m_s2
        self._bias_walk = accel_bias_walk_m_s2_sqrt_s
        self._nis_limit = gate_nis
        self._bias_limit = max_estimated_accel_bias_m_s2
        self._x = np.zeros(6, dtype=np.float64)
        self._p = np.diag([1., 1., 0.25, 0.25, 0.04, 0.04])
        self._last_imu: ImuReading | None = None
        self._last_visual_time_us: int | None = None
        self._visual_updates = 0
        self._last_nis = float("nan")
        self._status = "WAITING_FOR_IMU"

    def snapshot(self) -> FusionOutput:
        return FusionOutput(
            self._last_imu.timestamp_us if self._last_imu else 0,
            self._status, *(float(v) for v in self._x),
            self._last_nis, self._visual_updates,
            tuple(tuple(float(v) for v in row) for row in self._p),
        )

    def predict(self, reading: ImuReading) -> FusionOutput:
        # Validates readings and latches lost-history failure in preintegrator.
        q0 = self._imu.q.copy()
        try:
            delta = self._imu.step(reading)
        except ValueError:
            self._status = "IMU_INVALID_OR_GAP"
            return self.snapshot()
        if self._last_imu is None:
            self._last_imu = reading
            self._status = "INERTIAL_ONLY"
            return self.snapshot()
        dt = delta.dt_s
        q1 = self._imu.q.copy()
        q_mid = q0 + (q1 if q0 @ q1 >= 0 else -q1)
        q_mid = q_mid / np.linalg.norm(q_mid)
        r = rotation_body_to_ned(q_mid)
        f0 = _vector(self._last_imu.specific_force_body_m_s2, 3, "ускорение")
        f1 = _vector(reading.specific_force_body_m_s2, 3, "ускорение")
        specific_force = 0.5 * (f0 + f1) - self._imu.ba
        specific_force[:2] -= self._x[4:6]
        acceleration = (r @ specific_force + GRAVITY_NED)[:2]

        # Discrete dynamics for p,v and additive accelerometer-bias states.
        self._x[:2] += self._x[2:4] * dt + 0.5 * acceleration * dt * dt
        self._x[2:4] += acceleration * dt
        f = np.eye(6)
        f[:2, 2:4] = np.eye(2) * dt
        f[:2, 4:6] = -r[:2, :2] * (0.5 * dt * dt)
        f[2:4, 4:6] = -r[:2, :2] * dt

        g = np.zeros((6, 2))
        g[:2, :] = r[:2, :2] * (0.5 * dt * dt)
        g[2:4, :] = r[:2, :2] * dt
        q = (self._accel_noise ** 2) * (g @ g.T)
        q[4:6, 4:6] += np.eye(2) * self._bias_walk ** 2 * dt
        # Conservative additional horizontal error from gyro uncertainty
        # is NOT claimed here; gyro error is tracked as an unmodeled source.
        self._p = f @ self._p @ f.T + q
        self._p = 0.5 * (self._p + self._p.T)
        self._last_imu = reading
        if not np.isfinite(self._x).all() or not np.isfinite(self._p).all():
            self._status = "NUMERICAL_FAILURE"
            return self.snapshot()
        self._status = "INERTIAL_ONLY"
        return self.snapshot()

    def update_visual(self, measurement: VisualVelocityMeasurement) -> FusionOutput:
        if self._status in ("IMU_INVALID_OR_GAP", "NUMERICAL_FAILURE",
                            "ACCEL_BIAS_OUT_OF_BOUNDS"):
            return self.snapshot()
        if self._last_imu is None:
            self._status = "NO_IMU_REFERENCE"
            return self.snapshot()
        # Exact time-alignment until a delayed-state/repropagation scheme exists.
        if not isinstance(measurement.timestamp_us, int) or (
            measurement.timestamp_us != self._last_imu.timestamp_us
        ):
            self._status = "UNALIGNED_VISUAL_TIMESTAMP"
            return self.snapshot()
        if (self._last_visual_time_us is not None and
                measurement.timestamp_us <= self._last_visual_time_us):
            self._status = "REPEATED_VISUAL_MEASUREMENT"
            return self.snapshot()
        z = np.asarray([measurement.north_m_s, measurement.east_m_s], dtype=float)
        r = np.asarray(measurement.covariance_ne_m2_s2, dtype=float)
        if z.shape != (2,) or not np.isfinite(z).all() or r.shape != (2, 2) or (
            not np.isfinite(r).all()
        ) or not np.allclose(r, r.T, atol=1e-8) or (
            np.linalg.eigvalsh(r)[0] <= 0
        ):
            self._status = "VISUAL_COVARIANCE_INVALID"
            return self.snapshot()
        h = np.zeros((2, 6))
        h[:, 2:4] = np.eye(2)
        residual = z - h @ self._x
        s = h @ self._p @ h.T + r
        try:
            # Cholesky is required for SPD innovation covariance.
            chol = np.linalg.cholesky(s)
            whitened = np.linalg.solve(chol, residual)
            nis = float(whitened @ whitened)
        except np.linalg.LinAlgError:
            self._status = "INNOVATION_COVARIANCE_INVALID"
            return self.snapshot()
        self._last_nis = nis
        self._last_visual_time_us = measurement.timestamp_us
        if nis > self._nis_limit:
            self._status = "VISUAL_OUTLIER_REJECTED"
            return self.snapshot()
        gain = np.linalg.solve(s, h @ self._p).T
        proposed_x = self._x + gain @ residual
        if np.max(np.abs(proposed_x[4:6])) > self._bias_limit:
            self._status = "ACCEL_BIAS_OUT_OF_BOUNDS"
            return self.snapshot()
        i_kh = np.eye(6) - gain @ h
        proposed_p = i_kh @ self._p @ i_kh.T + gain @ r @ gain.T
        proposed_p = 0.5 * (proposed_p + proposed_p.T)
        if not np.isfinite(proposed_p).all() or (
            np.linalg.eigvalsh(proposed_p)[0] <= 0
        ):
            self._status = "POSTERIOR_COVARIANCE_INVALID"
            return self.snapshot()
        self._x, self._p = proposed_x, proposed_p
        self._visual_updates += 1
        self._status = "VISUAL_CORRECTED"
        return self.snapshot()
