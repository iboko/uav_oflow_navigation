"""Офлайн-сопровождение последовательности кадров нижней камеры БВС.

Результат: относительное смещение внутри непрерывного наблюдаемого участка.
Ни абсолютное положение, ни VIО, ни коррекция по карте не реализованы.
При пропуске/потере трекинга начинается новый независимый участок с нулевым
началом координат: запрещено сшивать их без внешнего источника положения.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, hypot, isfinite, pi, sin, sqrt

import numpy as np

from .ground_visual_motion import CameraCalibration, GroundMotionEstimate, estimate_ground_motion
from .camera_rotation_compensation import (CameraMountCalibration, align_current_to_previous,
                                           camera_delta_to_center_ned)
from .velocity_uncertainty import VelocityErrorAssumptions, modeled_velocity_covariance_ned


def _wrap(angle: float) -> float:
    return atan2(sin(angle), cos(angle))


@dataclass(frozen=True)
class VisualFrame:
    """Временная метка СЕРЕДИНЫ экспозиции (общая шкала времени, мкс)."""

    timestamp_us: int
    gray: np.ndarray
    height_agl_m: float
    yaw_rad: float
    roll_rad: float = 0.0
    pitch_rad: float = 0.0


@dataclass(frozen=True)
class VisualOdomResult:
    timestamp_us: int
    segment_id: int
    status: str
    accepted: bool
    north_m: float = 0.0
    east_m: float = 0.0
    vn_m_s: float = float("nan")
    ve_m_s: float = float("nan")
    inlier_count: int = 0
    inlier_ratio: float = 0.0
    # Только ориентировочный накопленный предел разрешения пикселей; не
    # доверительный интервал, не сертифицированная погрешность положения.
    resolution_proxy_m: float = 0.0
    velocity_cov_nn: float = float("nan")
    velocity_cov_ne: float = float("nan")
    velocity_cov_ee: float = float("nan")


class ContinuousPlanarOdometry:
    """Относительная визуальная одометрия для надирной камеры, офлайн.

    Обязательные условия: камера с известными внутренними параметрами,
    согласованные высота и курс, практически плоская неподвижная местность,
    ограниченные крен/тангаж и угловое движение. Все результаты являются
    ОТНОСИТЕЛЬНЫМИ, каждый разрыв дает новый независимый segment_id.
    """

    def __init__(
        self,
        calibration: CameraCalibration,
        *,
        camera_mount: CameraMountCalibration | None = None,
        camera_offset_body_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
        velocity_error_assumptions: VelocityErrorAssumptions | None = None,
        max_pair_dt_s: float = 0.35,
        max_yaw_change_rad: float = 0.30,
        max_relative_height_change: float = 0.12,
        max_tilt_rad: float = 0.15,
        max_delta_tilt_rad: float = 0.003,
        min_inliers: int = 20,
        min_inlier_ratio: float = 0.60,
        pixel_resolution_floor: float = 0.5,
    ) -> None:
        if not calibration.valid():
            raise ValueError("Неверные параметры камеры или ориентации ее осей")
        if (
            not 0.0 < max_pair_dt_s <= 1.0
            or not 0.0 < max_yaw_change_rad <= pi
            or not 0.0 < max_relative_height_change < 1.0
            or not 0.0 < max_tilt_rad <= 0.4
            or not 0.0 < max_delta_tilt_rad <= 0.10
            or min_inliers < 5 or not 0.0 < min_inlier_ratio <= 1.0
            or not isfinite(pixel_resolution_floor) or pixel_resolution_floor <= 0.0
        ):
            raise ValueError("Неверные параметры контроля визуальной одометрии")
        if camera_mount is not None and not camera_mount.valid(calibration):
            raise ValueError("Несогласованная пространственная калибровка камеры")
        offset = np.asarray(camera_offset_body_m, dtype=np.float64)
        if offset.shape != (3,) or not np.isfinite(offset).all():
            raise ValueError("Плечо установки камеры должно быть конечным 3D-вектором")
        if float(np.linalg.norm(offset)) > 5.0:
            raise ValueError("Плечо камеры слишком велико")
        if camera_mount is None and np.any(offset != 0.0):
            raise ValueError("Ненулевое плечо требует трехмерной калибровки камеры")
        if (velocity_error_assumptions is not None and
                not velocity_error_assumptions.valid()):
            raise ValueError("Недопустимые предположения о погрешностях")
        self._velocity_errors = velocity_error_assumptions
        self._offset = tuple(float(v) for v in offset)
        self._calib = calibration
        self._mount = camera_mount
        self._max_pair_dt = max_pair_dt_s
        self._max_yaw = max_yaw_change_rad
        self._max_height_jump = max_relative_height_change
        self._max_tilt = max_tilt_rad
        self._max_delta_tilt = max_delta_tilt_rad
        self._min_inliers = min_inliers
        self._min_ratio = min_inlier_ratio
        self._pixel_resolution = pixel_resolution_floor
        self._previous: VisualFrame | None = None
        self._segment_id = 0
        self._north = 0.0
        self._east = 0.0
        self._resolution = 0.0

    def reset(self) -> None:
        """Явный разрыв относительной траектории."""
        self._restart(None)

    def reject_frame(self, timestamp_us: int, reason: str) -> VisualOdomResult:
        """Явно зарегистрировать отказ внешнего канала измерений."""
        self._restart(None)
        return self._out(timestamp_us, reason)

    def _restart(self, anchor: VisualFrame | None) -> None:
        self._segment_id += 1
        self._previous = anchor
        self._north = 0.0
        self._east = 0.0
        self._resolution = 0.0

    def _out(self, timestamp: int, status: str, accepted: bool = False,
             motion: GroundMotionEstimate | None = None,
             vn: float = float("nan"), ve: float = float("nan"),
             covariance: np.ndarray | None = None) -> VisualOdomResult:
        return VisualOdomResult(
            timestamp, self._segment_id, status, accepted,
            self._north, self._east, vn, ve,
            motion.inlier_points if motion else 0,
            motion.inlier_ratio if motion else 0.0,
            self._resolution,
            float(covariance[0, 0]) if covariance is not None else float("nan"),
            float(covariance[0, 1]) if covariance is not None else float("nan"),
            float(covariance[1, 1]) if covariance is not None else float("nan"),
        )

    @staticmethod
    def _frame_valid(frame: VisualFrame) -> bool:
        return (
            isinstance(frame.timestamp_us, int) and frame.timestamp_us > 0
            and isinstance(frame.gray, np.ndarray)
            and frame.gray.ndim == 2 and frame.gray.dtype == np.uint8
            and min(frame.gray.shape) >= 80
            and all(isfinite(v) for v in (
                frame.height_agl_m, frame.yaw_rad,
                frame.roll_rad, frame.pitch_rad
            ))
            and 0.0 < frame.height_agl_m <= 150.0
        )

    @staticmethod
    def _snapshot(frame: VisualFrame) -> VisualFrame:
        # Camera drivers may reuse frame buffers between calls.
        return VisualFrame(frame.timestamp_us, frame.gray.copy(), frame.height_agl_m,
                           frame.yaw_rad, frame.roll_rad, frame.pitch_rad)

    def process(self, frame: VisualFrame) -> VisualOdomResult:
        if not self._frame_valid(frame):
            self._restart(None)
            return self._out(frame.timestamp_us, "INVALID_FRAME")

        current = self._snapshot(frame)
        if self._previous is None:
            self._previous = current
            return self._out(current.timestamp_us, "INITIALIZED")

        previous = self._previous
        if current.timestamp_us <= previous.timestamp_us:
            # A nonmonotonic timestamp compromises the measurement interval;
            # do not use this frame as the next reference.
            self._restart(None)
            return self._out(current.timestamp_us, "NON_MONOTONIC_TIME")

        dt_s = (current.timestamp_us - previous.timestamp_us) * 1e-6
        if dt_s > self._max_pair_dt:
            self._restart(current)
            return self._out(current.timestamp_us, "TIME_GAP_REINITIALIZED")

        if current.gray.shape != previous.gray.shape:
            self._restart(current)
            return self._out(current.timestamp_us, "IMAGE_SIZE_CHANGED")

        yaw_delta = _wrap(current.yaw_rad - previous.yaw_rad)
        if abs(yaw_delta) > self._max_yaw:
            self._restart(current)
            return self._out(current.timestamp_us, "EXCESSIVE_YAW_CHANGE")

        if (
            abs(current.height_agl_m - previous.height_agl_m)
            / max(current.height_agl_m, previous.height_agl_m)
            > self._max_height_jump
        ):
            self._restart(current)
            return self._out(current.timestamp_us, "HEIGHT_JUMP")

        # Both images must be within the validity domain of the planar model.
        if (
            hypot(previous.roll_rad, previous.pitch_rad) > self._max_tilt
            or hypot(current.roll_rad, current.pitch_rad) > self._max_tilt
        ):
            self._restart(current)
            return self._out(current.timestamp_us, "TILT_EXCEEDS_PLANAR_MODEL")

        # Without full 3D camera/IMU calibration, a changing roll/pitch angle
        # creates image translation indistinguishable from vehicle motion.
        # Reject it rather than synthesizing false horizontal displacement.
        if self._mount is None and hypot(
            current.roll_rad - previous.roll_rad,
            current.pitch_rad - previous.pitch_rad,
        ) > self._max_delta_tilt:
            self._restart(current)
            return self._out(current.timestamp_us, "UNCOMPENSATED_TILT_CHANGE")

        # With a calibrated camera/body SO(3) matrix, remove the known
        # rotational homography before estimating scene translation.
        # In the absence of that calibration retain conservative gating.
        corrected_current = current.gray
        if self._mount is not None:
            try:
                corrected_current, _ = align_current_to_previous(
                    current.gray, self._calib, self._mount,
                    previous_rpy=(previous.roll_rad, previous.pitch_rad,
                                  previous.yaw_rad),
                    current_rpy=(current.roll_rad, current.pitch_rad,
                                 current.yaw_rad),
                )
            except (ValueError, np.linalg.LinAlgError):
                self._restart(current)
                return self._out(current.timestamp_us, "ROTATION_COMPENSATION_FAILED")

        height = 0.5 * (previous.height_agl_m + current.height_agl_m)
        motion = estimate_ground_motion(
            previous.gray, corrected_current, dt_s, height,
            self._calib, roll_rad=previous.roll_rad, pitch_rad=previous.pitch_rad,
            max_tilt_rad=self._max_tilt, min_inliers=self._min_inliers,
            min_inlier_ratio=self._min_ratio,
        )
        if not motion.valid:
            # The new frame can seed a new segment but cannot bridge missing
            # displacement to the old segment.
            self._restart(current)
            return self._out(current.timestamp_us, f"TRACK_LOST:{motion.reason}", motion=motion)

        # Derotated visual displacement is expressed in PREVIOUS body axes.
        # Correct for rotation of the rigid camera lever arm before treating
        # camera displacement as centre-of-mass displacement.
        if self._mount is not None:
            try:
                dn_de = camera_delta_to_center_ned(
                    (motion.velocity_body_x_m_s * dt_s,
                     motion.velocity_body_y_m_s * dt_s),
                    (previous.roll_rad, previous.pitch_rad, previous.yaw_rad),
                    (current.roll_rad, current.pitch_rad, current.yaw_rad),
                    self._offset,
                )
            except ValueError:
                self._restart(current)
                return self._out(current.timestamp_us, "LEVER_ARM_CORRECTION_FAILED")
            vn, ve = float(dn_de[0] / dt_s), float(dn_de[1] / dt_s)
        else:
            mid_yaw = previous.yaw_rad + 0.5 * yaw_delta
            c, s = cos(mid_yaw), sin(mid_yaw)
            vn = c * motion.velocity_body_x_m_s - s * motion.velocity_body_y_m_s
            ve = s * motion.velocity_body_x_m_s + c * motion.velocity_body_y_m_s
        if not isfinite(vn) or not isfinite(ve):
            self._restart(current)
            return self._out(current.timestamp_us, "NONFINITE_VELOCITY")
        covariance = None
        if self._velocity_errors is not None:
            try:
                covariance = modeled_velocity_covariance_ned(
                    self._velocity_errors, camera=self._calib, motion=motion,
                    height_agl_m=height, interval_s=dt_s,
                    previous_rpy=(previous.roll_rad, previous.pitch_rad,
                                  previous.yaw_rad),
                    current_rpy=(current.roll_rad, current.pitch_rad,
                                 current.yaw_rad),
                    center_velocity_ned_m_s=(vn, ve),
                )
            except ValueError:
                self._restart(current)
                return self._out(current.timestamp_us, "UNCERTAINTY_MODEL_FAILED")
        self._north += vn * dt_s
        self._east += ve * dt_s

        # This is a resolution indicator, NOT an estimated covariance:
        # systematic height, calibration and map biases are not accounted for.
        du_floor = height * self._pixel_resolution / self._calib.focal_x_px
        dv_floor = height * self._pixel_resolution / self._calib.focal_y_px
        self._resolution += sqrt(du_floor * du_floor + dv_floor * dv_floor)
        self._previous = current
        return self._out(current.timestamp_us, "VALID", accepted=True,
                         motion=motion, vn=vn, ve=ve, covariance=covariance)
