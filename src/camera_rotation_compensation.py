"""Геометрическая компенсация поворота жестко установленной нижней камеры.

Расчет для калиброванной идеальной камеры без дисторсии. Выполняется только
в исследовательском контуре; не заменяет предынтегрирование ИИМ или VIO.
Оси: связанная СК БВС FRD (X вперед, Y вправо, Z вниз);
камера: X вправо изображения, Y вниз изображения, Z оптическая ось.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin, isfinite

import cv2
import numpy as np

from .ground_visual_motion import CameraCalibration


@dataclass(frozen=True)
class CameraMountCalibration:
    """Матрица перехода из СК камеры в связанную СК БВС, 3x3."""

    body_from_camera: tuple[tuple[float, float, float], ...]

    def valid(self, camera: CameraCalibration) -> bool:
        if not camera.valid():
            return False
        rotation = np.asarray(self.body_from_camera, dtype=np.float64)
        if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
            return False
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3):
            return False
        if abs(float(np.linalg.det(rotation)) - 1.0) > 1e-3:
            return False
        # The planar model is intended strictly for an almost downward camera.
        if float(rotation[2, 2]) < 0.985:
            return False
        # The 2D conversion must be consistent with the mounted 3D axes.
        return bool(np.allclose(rotation[:2, :2],
                                np.asarray(camera.image_to_body_xy),
                                atol=1e-3))


def body_to_ned(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """R_nb = Rz(yaw) Ry(pitch) Rx(roll), for NED and FRD axes."""
    if not all(isfinite(v) for v in (roll, pitch, yaw)):
        raise ValueError("Неконечная ориентация БВС")
    cr, sr = cos(roll), sin(roll)
    cp, sp = cos(pitch), sin(pitch)
    cy, sy = cos(yaw), sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=np.float64)


def rotation_homography_previous_to_current(
    camera: CameraCalibration,
    mount: CameraMountCalibration,
    *,
    previous_rpy: tuple[float, float, float],
    current_rpy: tuple[float, float, float],
) -> np.ndarray:
    """Пиксельная гомография чистого вращения предыдущий кадр -> текущий.

    R_c2_c1 = R_bc^T R_nb2^T R_nb1 R_bc.
    H = K R_c2_c1 K^-1.
    """
    if not mount.valid(camera):
        raise ValueError("Не согласованы калибровки установки камеры")
    previous_nb = body_to_ned(*previous_rpy)
    current_nb = body_to_ned(*current_rpy)
    rotation_bc = np.asarray(mount.body_from_camera, dtype=np.float64)
    rotation_c2_c1 = rotation_bc.T @ current_nb.T @ previous_nb @ rotation_bc
    k = np.array([
        [camera.focal_x_px, 0.0, camera.center_x_px],
        [0.0, camera.focal_y_px, camera.center_y_px],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    h = k @ rotation_c2_c1 @ np.linalg.inv(k)
    if not np.isfinite(h).all() or abs(h[2, 2]) < 1e-10:
        raise ValueError("Вырожденная гомография вращения")
    return h / h[2, 2]


def align_current_to_previous(
    current_gray: np.ndarray,
    camera: CameraCalibration,
    mount: CameraMountCalibration,
    *,
    previous_rpy: tuple[float, float, float],
    current_rpy: tuple[float, float, float],
    min_overlap_fraction: float = 0.80,
) -> tuple[np.ndarray, float]:
    """Разворачивает только угловое движение, сохраняет поступательное.

    Возвращает изображение в ориентации предыдущего кадра и долю видимой
    области. Пустые области заполняются нулем; при малом перекрытии отказ.
    """
    if not isinstance(current_gray, np.ndarray) or current_gray.ndim != 2:
        raise ValueError("Требуется одноканальный кадр камеры")
    if current_gray.dtype != np.uint8:
        raise ValueError("Требуются изображения uint8 после калибровки")
    if not 0.0 < min_overlap_fraction <= 1.0:
        raise ValueError("Недопустимый порог перекрытия")
    h_prev_curr = rotation_homography_previous_to_current(
        camera, mount, previous_rpy=previous_rpy, current_rpy=current_rpy
    )
    h_curr_prev = np.linalg.inv(h_prev_curr)
    height, width = current_gray.shape
    output_size = (width, height)
    aligned = cv2.warpPerspective(
        current_gray, h_curr_prev, output_size,
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    visible = cv2.warpPerspective(
        np.full_like(current_gray, 255), h_curr_prev, output_size,
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    overlap = float(np.count_nonzero(visible)) / visible.size
    if overlap < min_overlap_fraction:
        raise ValueError("Слишком малое перекрытие кадров после компенсации")
    return aligned, overlap
