"""Оценка горизонтального движения по последовательности кадров нижней камеры.

Экспериментальный наземный контур; НЕ готовая визуальная одометрия БВС.
Допущения: плоская статичная сцена, камера близка к надиру, известна высота,
известна калибровка и ориентация осей изображения относительно корпуса.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraCalibration:
    focal_x_px: float
    focal_y_px: float
    center_x_px: float
    center_y_px: float
    # Two row-major lines: pixel-plane metric XY -> body FRD horizontal XY.
    # Must be determined by mount/calibration; never assume identity in flight.
    image_to_body_xy: Tuple[Tuple[float, float], Tuple[float, float]]

    def valid(self) -> bool:
        rotation = np.asarray(self.image_to_body_xy, dtype=float)
        return (
            all(isfinite(v) for v in (
                self.focal_x_px, self.focal_y_px, self.center_x_px, self.center_y_px
            ))
            and self.focal_x_px > 0.0 and self.focal_y_px > 0.0
            and rotation.shape == (2, 2)
            and np.isfinite(rotation).all()
            and np.allclose(rotation.T @ rotation, np.eye(2), atol=1e-3)
            and abs(abs(float(np.linalg.det(rotation))) - 1.0) < 1e-3
        )


@dataclass(frozen=True)
class GroundMotionEstimate:
    valid: bool
    reason: str
    velocity_body_x_m_s: float = float("nan")
    velocity_body_y_m_s: float = float("nan")
    displacement_u_px: float = float("nan")
    displacement_v_px: float = float("nan")
    tracked_points: int = 0
    inlier_points: int = 0
    inlier_ratio: float = 0.0


def estimate_ground_motion(
    previous_image: np.ndarray,
    current_image: np.ndarray,
    dt_s: float,
    height_agl_m: float,
    calibration: CameraCalibration,
    *,
    roll_rad: float = 0.0,
    pitch_rad: float = 0.0,
    max_tilt_rad: float = 0.15,
    min_inliers: int = 20,
    min_inlier_ratio: float = 0.60,
    max_horizontal_speed_m_s: float = 15.0,
) -> GroundMotionEstimate:
    """LK tracking + forward/backward test + RANSAC partial-affine plane motion.

    Affine map moves terrain features from previous to current image.
    Translation at the calibrated principal point is converted to motion of
    the camera with the opposite sign; 2D orientation is an explicit calibration.
    No absolute coordinates, VIO or map matching are implied.
    """
    def reject(reason: str, tracked: int = 0, inliers: int = 0) -> GroundMotionEstimate:
        return GroundMotionEstimate(
            False, reason, tracked_points=tracked, inlier_points=inliers,
            inlier_ratio=inliers / tracked if tracked else 0.0
        )

    if not calibration.valid():
        return reject("INVALID_CALIBRATION")
    if not all(map(isfinite, (dt_s, height_agl_m, roll_rad, pitch_rad))):
        return reject("NON_FINITE_INPUT")
    if dt_s <= 0.0 or dt_s > 0.5 or height_agl_m <= 0.0 or height_agl_m > 150.0:
        return reject("INVALID_TIME_OR_HEIGHT")
    if abs(roll_rad) > max_tilt_rad or abs(pitch_rad) > max_tilt_rad:
        return reject("TILT_EXCEEDS_PLANAR_MODEL")
    if (
        previous_image.ndim != 2 or current_image.ndim != 2
        or previous_image.dtype != np.uint8 or current_image.dtype != np.uint8
        or previous_image.shape != current_image.shape
        or min(previous_image.shape) < 80
    ):
        return reject("INVALID_IMAGE")

    initial = cv2.goodFeaturesToTrack(
        previous_image, maxCorners=800, qualityLevel=0.012,
        minDistance=8.0, blockSize=7
    )
    if initial is None or len(initial) < min_inliers:
        return reject("INSUFFICIENT_TEXTURE")

    current, forward_status, _ = cv2.calcOpticalFlowPyrLK(
        previous_image, current_image, initial, None, winSize=(21, 21),
        maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )
    if current is None or forward_status is None:
        return reject("TRACKING_FAILED")
    previous_back, backward_status, _ = cv2.calcOpticalFlowPyrLK(
        current_image, previous_image, current, None, winSize=(21, 21),
        maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )
    if previous_back is None or backward_status is None:
        return reject("TRACKING_FAILED")

    old = initial.reshape(-1, 2)
    new = current.reshape(-1, 2)
    back = previous_back.reshape(-1, 2)
    forward_ok = forward_status.reshape(-1) == 1
    backward_ok = backward_status.reshape(-1) == 1
    fb_error = np.linalg.norm(old - back, axis=1)
    good = (
        forward_ok & backward_ok & np.isfinite(new).all(axis=1)
        & np.isfinite(back).all(axis=1) & (fb_error < 0.8)
    )
    old, new = old[good], new[good]
    tracked = len(old)
    if tracked < min_inliers:
        return reject("TOO_FEW_RELIABLE_TRACKS", tracked)

    transform, mask = cv2.estimateAffinePartial2D(
        old, new, method=cv2.RANSAC, ransacReprojThreshold=1.8,
        maxIters=2000, confidence=0.995, refineIters=10
    )
    if transform is None or mask is None or not np.isfinite(transform).all():
        return reject("GEOMETRIC_MODEL_FAILED", tracked)
    n_inlier = int(mask.reshape(-1).sum())
    if n_inlier < min_inliers or n_inlier / tracked < min_inlier_ratio:
        return reject("INSUFFICIENT_INLIERS", tracked, n_inlier)

    # A cluster of points in a tiny image region cannot reliably separate
    # image translation, scale and rotation over the full field of view.
    inliers = old[mask.reshape(-1).astype(bool)]
    height_px, width_px = previous_image.shape
    cells_x = np.clip((inliers[:, 0] * 4 / width_px).astype(int), 0, 3)
    cells_y = np.clip((inliers[:, 1] * 4 / height_px).astype(int), 0, 3)
    spatial_cells = len(set(zip(cells_x.tolist(), cells_y.tolist())))
    if spatial_cells < 6:
        return reject("INSUFFICIENT_SPATIAL_DISTRIBUTION", tracked, n_inlier)

    projected = inliers @ transform[:, :2].T + transform[:, 2]
    target = new[mask.reshape(-1).astype(bool)]
    residual = np.linalg.norm(projected - target, axis=1)
    if not np.isfinite(residual).all() or float(np.median(residual)) > 1.25:
        return reject("EXCESSIVE_REPROJECTION_ERROR", tracked, n_inlier)

    principal = np.array([calibration.center_x_px, calibration.center_y_px])
    pixel_shift = transform[:, :2] @ principal + transform[:, 2] - principal
    image_plane_speed = np.array([
        -pixel_shift[0] * height_agl_m / (calibration.focal_x_px * dt_s),
        -pixel_shift[1] * height_agl_m / (calibration.focal_y_px * dt_s),
    ])
    velocity_body = np.asarray(calibration.image_to_body_xy) @ image_plane_speed
    if not np.isfinite(velocity_body).all():
        return reject("NUMERICAL_FAILURE", tracked, n_inlier)
    if np.linalg.norm(velocity_body) > max_horizontal_speed_m_s:
        return reject("SPEED_EXCEEDS_LIMIT", tracked, n_inlier)

    return GroundMotionEstimate(
        True, "OK", float(velocity_body[0]), float(velocity_body[1]),
        float(pixel_shift[0]), float(pixel_shift[1]), tracked, n_inlier,
        n_inlier / tracked
    )
