"""Привязка к ортофотоплану другого года: ортотрансформирование + NCC.

Признаки ORB/SIFT между съемками разных лет и сезонов (тени, листва,
машины, стройки) не сопоставляются: на SWISSIMAGE 2019 против 2025
SIFT дал 0–5 точек. Устойчивее корреляция областей. Кадр
ортотрансформируется на ровную землю по ориентации автопилота и
дальномеру в масштабе карты (север вверх), затем нормированная
взаимная корреляция (TM_CCOEFF_NORMED) ищет его в окне карты вокруг
прогноза положения.

Принятие (пороги заданы до испытаний):
- пик ≥ ``min_peak``;
- пик − второй пик (вне круга ``exclusion_m``) ≥ ``min_margin`` —
  защита от повторяющихся структур;
- патч целиком в кадре (иначе наклон слишком велик).

Курс не измеряется: ошибка курса автопилота в несколько градусов
ослабляет пик, но не смещает решение заметно. Ковариация — только из
калибровки по отдельным полетам.
"""
from __future__ import annotations

from math import isfinite

import cv2
import numpy as np

from .camera_rotation_compensation import CameraMountCalibration, body_to_ned
from .ground_visual_motion import CameraCalibration
from .orthophoto_localization import MapGeoReference, MapLocalizationFix


class OrthoNccLocalizer:
    def __init__(self, gray: np.ndarray, georef: MapGeoReference, calibration: CameraCalibration,
                 mount: CameraMountCalibration, *, camera_offset_body_m=(0.0, 0.0, 0.0),
                 patch_m: float = 30.0, min_peak: float = 0.35, min_margin: float = 0.05,
                 exclusion_m: float = 3.0, max_tilt_rad: float = 0.12,
                 yaw_search_deg=(-6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0)) -> None:
        jac = georef.jacobian()
        if not (abs(jac[0, 0]) < 1e-9 and abs(jac[1, 1]) < 1e-9 and jac[0, 1] < 0 and jac[1, 0] > 0
                and abs(-jac[0, 1] - jac[1, 0]) < 1e-9):
            raise ValueError("Нужна карта север-вверх с квадратным пикселем")
        if not mount.valid(calibration):
            raise ValueError("Несогласованная калибровка камеры")
        self.gsd = float(jac[1, 0])
        self.origin = np.asarray(georef.origin_ne_m, dtype=float)
        self.map = gray
        self.k = np.array([[calibration.focal_x_px, 0, calibration.center_x_px],
                           [0, calibration.focal_y_px, calibration.center_y_px], [0, 0, 1.0]])
        self.r_bc = np.asarray(mount.body_from_camera, dtype=float)
        self.offset = np.asarray(camera_offset_body_m, dtype=float)
        self.npx = int(round(patch_m / self.gsd))
        self.min_peak, self.min_margin = min_peak, min_margin
        self.exclusion_px = max(1, int(round(exclusion_m / self.gsd)))
        self.max_tilt = max_tilt_rad
        self.yaw_search = tuple(np.radians(yaw_search_deg))
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        self.last_peak = float("nan")
        self.last_margin = float("nan")

    def _pixel(self, n: float, e: float) -> tuple[float, float]:
        # n = origin_n - gsd*row ; e = origin_e + gsd*col
        return (e - self.origin[1]) / self.gsd, (self.origin[0] - n) / self.gsd

    def ortho_patch(self, frame: np.ndarray, height_agl_m: float, roll: float, pitch: float, yaw: float):
        """North-up patch centred below the camera; returns (patch, fully_inside)."""
        r_nc = body_to_ned(roll, pitch, yaw) @ self.r_bc
        js, is_ = np.meshgrid(np.arange(self.npx), np.arange(self.npx))
        dn = (self.npx / 2 - is_ - 0.5) * self.gsd
        de = (js - self.npx / 2 + 0.5) * self.gsd
        # Camera height above the ground: rangefinder height at the vehicle;
        # the camera offset is small against tens of metres (ignored here,
        # the lever arm is applied to the position below).
        pts = np.stack([dn, de, np.full_like(dn, height_agl_m)], -1)
        pc = pts @ r_nc
        uv = pc @ self.k.T
        mx = (uv[..., 0] / uv[..., 2]).astype(np.float32)
        my = (uv[..., 1] / uv[..., 2]).astype(np.float32)
        h, w = frame.shape
        inside = bool(np.all((mx >= 0) & (mx < w - 1) & (my >= 0) & (my < h - 1) & (pc[..., 2] > 0)))
        return cv2.remap(frame, mx, my, cv2.INTER_LINEAR), inside

    def localize(self, image: np.ndarray, *, prior_ne_m, radius_m: float, height_agl_m: float,
                 roll_rad: float, pitch_rad: float, yaw_rad: float) -> MapLocalizationFix:
        if not all(isfinite(x) for x in (height_agl_m, roll_rad, pitch_rad, yaw_rad, radius_m)):
            return MapLocalizationFix(False, "NONFINITE_INPUT")
        if prior_ne_m is None or not np.isfinite(prior_ne_m).all():
            return MapLocalizationFix(False, "NO_PRIOR_POSITION")
        if abs(roll_rad) > self.max_tilt or abs(pitch_rad) > self.max_tilt:
            return MapLocalizationFix(False, "UNSUPPORTED_ATTITUDE")
        if not 5.0 <= height_agl_m <= 150.0:
            return MapLocalizationFix(False, "UNSUPPORTED_HEIGHT")
        lever = (body_to_ned(roll_rad, pitch_rad, yaw_rad) @ self.offset)[:2]
        # The patch centre is the ground point below the CAMERA.
        cam_prior = np.asarray(prior_ne_m, dtype=float) + lever
        col, row = self._pixel(*cam_prior)
        rad = int(round(radius_m / self.gsd)) + self.npx // 2
        r0, c0 = int(round(row)) - rad, int(round(col)) - rad
        if r0 < 0 or c0 < 0 or r0 + 2 * rad > self.map.shape[0] or c0 + 2 * rad > self.map.shape[1]:
            return MapLocalizationFix(False, "OUTSIDE_MAP")
        window = self.clahe.apply(self.map[r0:r0 + 2 * rad, c0:c0 + 2 * rad])
        # Search over small heading offsets: with a 4 deg autopilot heading
        # error the correlation lost its margin in SITL. The heading is a
        # search nuisance here, not a measurement (it also absorbed a
        # systematic rotation between the 2019 and 2025 products).
        best = None
        for dyaw in self.yaw_search:
            patch, inside = self.ortho_patch(image, height_agl_m, roll_rad, pitch_rad, yaw_rad + dyaw)
            if not inside:
                return MapLocalizationFix(False, "PATCH_OUTSIDE_FRAME")
            patch = self.clahe.apply(patch)
            if float(np.std(patch)) < 5.0:
                return MapLocalizationFix(False, "INSUFFICIENT_TEXTURE")
            r = cv2.matchTemplate(window, patch, cv2.TM_CCOEFF_NORMED)
            _, pk, _, lc = cv2.minMaxLoc(r)
            if best is None or pk > best[0]:
                best = (pk, lc, r)
        peak, loc, res = best
        masked = res.copy()
        cv2.circle(masked, loc, self.exclusion_px, -1.0, -1)
        second = float(masked.max())
        self.last_peak, self.last_margin = float(peak), float(peak - second)
        if peak < self.min_peak:
            return MapLocalizationFix(False, "WEAK_CORRELATION")
        if peak - second < self.min_margin:
            return MapLocalizationFix(False, "AMBIGUOUS_MAP_MATCH")
        x, y = float(loc[0]), float(loc[1])
        # Sub-pixel peak: parabola through the neighbours.
        if 0 < loc[0] < res.shape[1] - 1:
            a, b, c = res[loc[1], loc[0] - 1], res[loc[1], loc[0]], res[loc[1], loc[0] + 1]
            den = a - 2 * b + c
            x += 0.5 * (a - c) / den if den < 0 else 0.0
        if 0 < loc[1] < res.shape[0] - 1:
            a, b, c = res[loc[1] - 1, loc[0]], res[loc[1], loc[0]], res[loc[1] + 1, loc[0]]
            den = a - 2 * b + c
            y += 0.5 * (a - c) / den if den < 0 else 0.0
        # The georeference origin is the centre of pixel (0, 0); the patch
        # centre (camera nadir) lies at index npx/2 - 0.5 inside the patch.
        cam_n = self.origin[0] - (r0 + y + self.npx / 2 - 0.5) * self.gsd
        cam_e = self.origin[1] + (c0 + x + self.npx / 2 - 0.5) * self.gsd
        north, east = cam_n - lever[0], cam_e - lever[1]
        return MapLocalizationFix(True, "NCC_ACCEPTED_UNVALIDATED", "ncc", float(north), float(east),
                                  0, int(round(100 * peak)), float(peak - second), float("nan"),
                                  float("nan"), False, float("nan"), float(cam_n), float(cam_e))
