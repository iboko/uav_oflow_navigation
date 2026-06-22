from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Tuple

import numpy as np

try:
    from config import NavigationConfig
except ImportError:
    from .config import NavigationConfig


@dataclass
class FlowVelocityResult:
    valid: bool
    vx_body_mps: float = math.nan
    vy_body_mps: float = math.nan
    vx_nav_mps: float = math.nan
    vy_nav_mps: float = math.nan
    height_m: float = math.nan
    reason: str = "OK"


def yaw_rotation_body_to_nav(vx_body: float, vy_body: float, yaw_rad: float) -> Tuple[float, float]:
    """Поворот горизонтальной скорости из связанной СК БВС в навигационную СК."""
    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)
    vx_nav = c * vx_body - s * vy_body
    vy_nav = s * vx_body + c * vy_body
    return vx_nav, vy_nav


def yaw_rotation_nav_to_body(vx_nav: float, vy_nav: float, yaw_rad: float) -> Tuple[float, float]:
    """Поворот горизонтальной скорости из навигационной СК в связанную СК БВС."""
    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)
    vx_body = c * vx_nav + s * vy_nav
    vy_body = -s * vx_nav + c * vy_nav
    return vx_body, vy_body


def altitude_from_range(range_m: float, roll_rad: float, pitch_rad: float) -> float:
    """Пересчет дальности вдоль луча дальномера в вертикальную высоту.

    Для малых углов: height ≈ range * cos(roll) * cos(pitch).
    """
    if not np.isfinite(range_m):
        return math.nan
    return float(range_m * math.cos(roll_rad) * math.cos(pitch_rad))


def optical_flow_to_velocity(
    integrated_x_rad: float,
    integrated_y_rad: float,
    integrated_xgyro_rad: float,
    integrated_ygyro_rad: float,
    dt_s: float,
    range_m: float,
    roll_rad: float,
    pitch_rad: float,
    yaw_rad: float,
    quality: float,
    cfg: NavigationConfig,
) -> FlowVelocityResult:
    """Преобразует MAVLink OPTICAL_FLOW_RAD + дальномер в горизонтальную скорость.

    Принятая модель для направленного вниз датчика:

        flow_x_rate = (integrated_x - integrated_xgyro) / dt
        flow_y_rate = (integrated_y - integrated_ygyro) / dt

        vx_body =  height * flow_y_rate
        vy_body = -height * flow_x_rate

    Здесь `integrated_xgyro` и `integrated_ygyro` используются для компенсации
    вращательного движения БВС. Если конкретный датчик уже выдает
    полностью компенсированный поток, компенсацию нужно отключить или
    изменить под его паспорт/драйвер.
    """
    vals = [
        integrated_x_rad,
        integrated_y_rad,
        integrated_xgyro_rad,
        integrated_ygyro_rad,
        dt_s,
        range_m,
        roll_rad,
        pitch_rad,
        yaw_rad,
        quality,
    ]
    if not all(np.isfinite(v) for v in vals):
        return FlowVelocityResult(False, reason="NON_FINITE_INPUT")

    if dt_s <= 0.0 or dt_s > 0.20:
        return FlowVelocityResult(False, reason="BAD_FLOW_DT")

    if quality < cfg.quality_min:
        return FlowVelocityResult(False, reason="LOW_FLOW_QUALITY")

    height_m = altitude_from_range(range_m, roll_rad, pitch_rad)
    if not np.isfinite(height_m):
        return FlowVelocityResult(False, reason="BAD_HEIGHT")
    if height_m < cfg.min_height_m:
        return FlowVelocityResult(False, height_m=height_m, reason="HEIGHT_TOO_LOW")
    if height_m > cfg.max_height_m:
        return FlowVelocityResult(False, height_m=height_m, reason="HEIGHT_TOO_HIGH")

    flow_x_rate = (integrated_x_rad - integrated_xgyro_rad) / dt_s
    flow_y_rate = (integrated_y_rad - integrated_ygyro_rad) / dt_s

    if abs(flow_x_rate) > cfg.max_abs_flow_rate_rad_s or abs(flow_y_rate) > cfg.max_abs_flow_rate_rad_s:
        return FlowVelocityResult(False, height_m=height_m, reason="FLOW_RATE_LIMIT")

    vx_body = height_m * flow_y_rate
    vy_body = -height_m * flow_x_rate

    speed = math.hypot(vx_body, vy_body)
    if speed > cfg.max_horizontal_speed_mps * 2.5:
        return FlowVelocityResult(False, vx_body, vy_body, height_m=height_m, reason="VELOCITY_IMPLAUSIBLE")

    vx_nav, vy_nav = yaw_rotation_body_to_nav(vx_body, vy_body, yaw_rad)
    return FlowVelocityResult(
        True,
        vx_body_mps=float(vx_body),
        vy_body_mps=float(vy_body),
        vx_nav_mps=float(vx_nav),
        vy_nav_mps=float(vy_nav),
        height_m=float(height_m),
        reason="OK",
    )


def velocity_to_optical_flow_rad(
    vx_nav_mps: float,
    vy_nav_mps: float,
    yaw_rad: float,
    height_m: float,
    dt_s: float,
    gyro_x_radps: float,
    gyro_y_radps: float,
    gyro_z_radps: float,
    flow_noise_rad: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float, float, float]:
    """Обратная модель для генерации синтетического OPTICAL_FLOW_RAD.

    Возвращает:
        integrated_x, integrated_y, integrated_xgyro, integrated_ygyro, integrated_zgyro
    """
    if rng is None:
        rng = np.random.default_rng()

    vx_body, vy_body = yaw_rotation_nav_to_body(vx_nav_mps, vy_nav_mps, yaw_rad)

    h = max(height_m, 0.05)
    flow_x_rate_trans = -vy_body / h
    flow_y_rate_trans = vx_body / h

    integrated_xgyro = gyro_x_radps * dt_s
    integrated_ygyro = gyro_y_radps * dt_s
    integrated_zgyro = gyro_z_radps * dt_s

    integrated_x = (flow_x_rate_trans + gyro_x_radps) * dt_s + rng.normal(0.0, flow_noise_rad)
    integrated_y = (flow_y_rate_trans + gyro_y_radps) * dt_s + rng.normal(0.0, flow_noise_rad)

    return (
        float(integrated_x),
        float(integrated_y),
        float(integrated_xgyro),
        float(integrated_ygyro),
        float(integrated_zgyro),
    )
