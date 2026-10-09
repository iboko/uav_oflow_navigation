"""Начальная модель ковариации горизонтальной скорости по изображению.

Составляется из ЗАДАННЫХ погрешностей калибровки и синхронизации.
До эмпирической проверки НЕ является статистически откалиброванной
ковариацией, доверительным интервалом или основанием для EKF2 fusion.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

from .camera_rotation_compensation import body_to_ned
from .ground_visual_motion import CameraCalibration, GroundMotionEstimate


@dataclass(frozen=True)
class VelocityErrorAssumptions:
    # Среднеквадратическая ошибка уже ОЦЕНЕННОГО смещения по изображению.
    # Не погрешность отдельной характерной точки; задается по экспериментам.
    fitted_displacement_sigma_px: float
    height_sigma_m: float
    interval_sigma_s: float
    yaw_sigma_rad: float
    focal_relative_sigma: float
    camera_offset_sigma_m: float = 0.0
    velocity_model_floor_m_s: float = 0.0

    def valid(self) -> bool:
        values = tuple(vars(self).values())
        return all(isfinite(x) and 0.0 <= x < 1000.0 for x in values) and (
            self.fitted_displacement_sigma_px > 0
            and self.height_sigma_m > 0
            and self.interval_sigma_s >= 0
            and self.yaw_sigma_rad >= 0
            and self.focal_relative_sigma >= 0
            and self.camera_offset_sigma_m >= 0
            and self.velocity_model_floor_m_s > 0
        )


def modeled_velocity_covariance_ned(
    assumptions: VelocityErrorAssumptions,
    *,
    camera: CameraCalibration,
    motion: GroundMotionEstimate,
    height_agl_m: float,
    interval_s: float,
    previous_rpy: tuple[float, float, float],
    current_rpy: tuple[float, float, float],
    center_velocity_ned_m_s: tuple[float, float],
) -> np.ndarray:
    """Линейное распространение ошибок измерений: J Sigma J^T.

    Предполагается независимость перечисленных погрешностей и плоская сцена.
    Ошибки рельефа, скользящего затвора, временная корреляция и дисторсия
    моделью НЕ учтены. Значения являются расчетными предположениями.
    """
    if not assumptions.valid() or not camera.valid() or not motion.valid:
        raise ValueError("Нет достоверных предположений для оценки ковариации")
    if not (isfinite(height_agl_m) and 0 < height_agl_m <= 150
            and isfinite(interval_s) and 0 < interval_s <= 0.5):
        raise ValueError("Недопустимая высота или временной интервал")
    v = np.asarray(center_velocity_ned_m_s, dtype=np.float64)
    if v.shape != (2,) or not np.isfinite(v).all():
        raise ValueError("Скорость не конечна")
    if not isfinite(motion.inlier_ratio) or not 0.0 < motion.inlier_ratio <= 1.0:
        raise ValueError("Недопустимая доля согласованных точек")

    r0 = body_to_ned(*previous_rpy)
    r1 = body_to_ned(*current_rpy)
    body_from_image = np.asarray(camera.image_to_body_xy, dtype=float)
    camera_vel_nav = r0[:2, :2] @ np.array([
        motion.velocity_body_x_m_s, motion.velocity_body_y_m_s
    ])
    # Image pixel displacement -> horizontal NED velocity.
    j_px = r0[:2, :2] @ body_from_image @ np.diag([
        -height_agl_m / (camera.focal_x_px * interval_s),
        -height_agl_m / (camera.focal_y_px * interval_s),
    ])
    # Inflated when inlier ratio drops; initial engineering assumption only.
    sigma_fit = assumptions.fitted_displacement_sigma_px / max(
        motion.inlier_ratio, 0.4
    )
    covariance = (j_px @ j_px.T) * (sigma_fit ** 2)
    covariance += np.outer(camera_vel_nav, camera_vel_nav) * (
        assumptions.height_sigma_m / height_agl_m
    ) ** 2
    covariance += np.outer(v, v) * (
        assumptions.interval_sigma_s / interval_s
    ) ** 2
    covariance += np.outer(camera_vel_nav, camera_vel_nav) * (
        assumptions.focal_relative_sigma ** 2
    )
    yaw_jac = np.array([-v[1], v[0]])
    covariance += np.outer(yaw_jac, yaw_jac) * assumptions.yaw_sigma_rad ** 2
    # Translational offset calibration uncertainty under finite attitude change.
    offset_jac = (r1[:2, :] - r0[:2, :]) / interval_s
    covariance += (offset_jac @ offset_jac.T) * assumptions.camera_offset_sigma_m ** 2
    covariance += np.eye(2) * assumptions.velocity_model_floor_m_s ** 2
    covariance = 0.5 * (covariance + covariance.T)
    if not np.isfinite(covariance).all() or np.linalg.eigvalsh(covariance)[0] <= 0:
        raise ValueError("Вырожденная или недопустимая расчетная ковариация")
    return covariance
