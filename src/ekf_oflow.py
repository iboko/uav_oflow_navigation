from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

try:
    from config import NavigationConfig
    from oflow_math import yaw_rotation_body_to_nav
except ImportError:
    from .config import NavigationConfig
    from .oflow_math import yaw_rotation_body_to_nav


@dataclass
class EkfState:
    t_s: float
    n_m: float
    e_m: float
    vn_mps: float
    ve_mps: float
    bax_mps2: float
    bay_mps2: float


class HorizontalOpticalFlowEKF:
    """Простой EKF горизонтальной навигации для первого этапа.

    Состояние:
        x = [n, e, vn, ve, bax, bay]^T

    где:
        n, e       — относительное положение;
        vn, ve     — горизонтальная скорость;
        bax, bay   — смещения акселерометров в связанной СК.

    Предсказание выполняется по горизонтальным ускорениям IMU,
    скорректированным по yaw. Коррекция выполняется по скорости,
    полученной из оптического потока и дальномера.
    """

    def __init__(self, cfg: NavigationConfig):
        self.cfg = cfg
        self.x = np.zeros(6, dtype=float)
        self.P = np.diag([0.20**2, 0.20**2, 0.12**2, 0.12**2, 0.03**2, 0.03**2])
        self.initialized = False
        self.last_t_s: float | None = None

    def initialize(self, t_s: float, n_m: float = 0.0, e_m: float = 0.0) -> None:
        self.x[:] = [n_m, e_m, 0.0, 0.0, 0.0, 0.0]
        self.initialized = True
        self.last_t_s = t_s

    def predict(self, t_s: float, acc_x_body_mps2: float, acc_y_body_mps2: float, yaw_rad: float) -> None:
        if not self.initialized:
            self.initialize(t_s)
            return

        dt = float(t_s - self.last_t_s)
        if dt <= 0.0:
            return
        if dt > 0.20:
            # При больших разрывах лучше не интегрировать рывком.
            dt = 0.20

        n, e, vn, ve, bax, bay = self.x

        ax_corr = acc_x_body_mps2 - bax
        ay_corr = acc_y_body_mps2 - bay
        an, ae = yaw_rotation_body_to_nav(ax_corr, ay_corr, yaw_rad)

        n_new = n + vn * dt + 0.5 * an * dt * dt
        e_new = e + ve * dt + 0.5 * ae * dt * dt
        vn_new = vn + an * dt
        ve_new = ve + ae * dt

        self.x = np.array([n_new, e_new, vn_new, ve_new, bax, bay], dtype=float)

        c = math.cos(yaw_rad)
        s = math.sin(yaw_rad)

        # a_nav = R(yaw) * (a_body - b_body)
        # d(a_nav)/d(bx,by) = -R(yaw)
        F = np.eye(6)
        F[0, 2] = dt
        F[1, 3] = dt
        F[0, 4] = -0.5 * c * dt * dt
        F[0, 5] = 0.5 * s * dt * dt
        F[1, 4] = -0.5 * s * dt * dt
        F[1, 5] = -0.5 * c * dt * dt
        F[2, 4] = -c * dt
        F[2, 5] = s * dt
        F[3, 4] = -s * dt
        F[3, 5] = -c * dt

        q_pos = 0.002 * dt
        q_vel = (self.cfg.accel_noise_sigma_mps2 * dt) ** 2
        q_bias = (0.002 * dt) ** 2
        Q = np.diag([q_pos, q_pos, q_vel, q_vel, q_bias, q_bias])

        self.P = F @ self.P @ F.T + Q
        self.P = 0.5 * (self.P + self.P.T)
        self.last_t_s = t_s

    def update_flow_velocity(
        self,
        vn_meas_mps: float,
        ve_meas_mps: float,
        quality: float,
        height_m: float,
    ) -> tuple[float, float]:
        """Коррекция по измерению горизонтальной скорости от ОП.

        Возвращает инновации по скорости.
        """
        if not self.initialized:
            raise RuntimeError("EKF не инициализирован")

        z = np.array([vn_meas_mps, ve_meas_mps], dtype=float)
        H = np.zeros((2, 6), dtype=float)
        H[0, 2] = 1.0
        H[1, 3] = 1.0

        q = float(np.clip(quality, 1.0, 255.0))
        # Чем ниже качество и выше высота, тем больше дисперсия скорости.
        sigma = self.cfg.base_flow_vel_sigma_mps
        sigma += 0.18 * (1.0 - q / 255.0)
        sigma += 0.025 * max(height_m - 1.0, 0.0)
        sigma = float(np.clip(sigma, 0.035, 0.35))

        R = np.diag([sigma * sigma, sigma * sigma])
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        I = np.eye(6)
        # Joseph form — устойчивее численно.
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        return float(y[0]), float(y[1])

    def state(self, t_s: float) -> EkfState:
        return EkfState(
            t_s=float(t_s),
            n_m=float(self.x[0]),
            e_m=float(self.x[1]),
            vn_mps=float(self.x[2]),
            ve_mps=float(self.x[3]),
            bax_mps2=float(self.x[4]),
            bay_mps2=float(self.x[5]),
        )
