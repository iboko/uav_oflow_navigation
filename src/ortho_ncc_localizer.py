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

Курс: лучший поворот из перебора (±12°, уточнение параболой) — это
курс по карте (``yaw_rad``), на краю перебора — NaN. Прежняя оценка
«систематический поворот −4° между съемками» была сделана при
артефактном старте SITL (ошибка наклона и курса EKF2). После
перезапуска EKF2 лучший поворот совпал с ошибкой курса автопилота.
Точность курса и ковариация положения — только из калибровки по
отдельным полетам.
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
                 yaw_search_deg=tuple(float(x) for x in range(-12, 13, 2)),
                 map_clahe_tile_m: float = 8.75) -> None:
        jac = georef.jacobian()
        if not (abs(jac[0, 0]) < 1e-9 and abs(jac[1, 1]) < 1e-9 and jac[0, 1] < 0 and jac[1, 0] > 0
                and abs(-jac[0, 1] - jac[1, 0]) < 1e-9):
            raise ValueError("Нужна карта север-вверх с квадратным пикселем")
        if not mount.valid(calibration):
            raise ValueError("Несогласованная калибровка камеры")
        offset = np.asarray(camera_offset_body_m, dtype=float)
        if offset.shape != (3,) or not np.isfinite(offset).all():
            raise ValueError("Некорректное смещение камеры")
        if (not isinstance(gray, np.ndarray) or gray.ndim != 2 or gray.dtype != np.uint8
                or not gray.size):
            raise ValueError("Нужна непустая карта uint8 в оттенках серого")
        if (not all(isfinite(x) for x in (patch_m, min_peak, min_margin, exclusion_m, max_tilt_rad))
                or patch_m <= 0 or not 0 <= min_peak <= 1 or not 0 < min_margin <= 2
                or exclusion_m <= 0 or not 0 < max_tilt_rad <= 0.2
                or not yaw_search_deg or not all(isfinite(x) for x in yaw_search_deg)):
            raise ValueError("Недопустимые параметры NCC")
        self.gsd = float(jac[1, 0])
        self.origin = np.asarray(georef.origin_ne_m, dtype=float)
        self.map = gray
        self.k = np.array([[calibration.focal_x_px, 0, calibration.center_x_px],
                           [0, calibration.focal_y_px, calibration.center_y_px], [0, 0, 1.0]])
        self.r_bc = np.asarray(mount.body_from_camera, dtype=float)
        self.offset = offset
        self.npx = int(round(patch_m / self.gsd))
        if self.npx < 3:
            raise ValueError("Патч NCC должен содержать не менее трех пикселей на сторону")
        self.min_peak, self.min_margin = min_peak, min_margin
        self.exclusion_px = max(1, int(round(exclusion_m / self.gsd)))
        self.max_tilt = max_tilt_rad
        self.yaw_search = tuple(np.radians(yaw_search_deg))
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        # The map is equalised once, with tiles of a fixed ground size. It
        # was equalised per search window: the result then depended on the
        # search radius (same place, 20 m window: peak 0.514, margin 0.086;
        # 10 m window after lock: 0.466 and 0.047 -> ambiguous; swiss test).
        # 8.75 m = the tile of the 20 m window used for the calibration.
        tiles = (max(1, int(round(gray.shape[0] * self.gsd / map_clahe_tile_m))),
                 max(1, int(round(gray.shape[1] * self.gsd / map_clahe_tile_m))))
        self.map_eq = cv2.createCLAHE(clipLimit=3.0, tileGridSize=tiles).apply(gray)
        self.last_peak = float("nan")
        self.last_margin = float("nan")

    def _pixel(self, n: float, e: float) -> tuple[float, float]:
        # n = origin_n - gsd*row ; e = origin_e + gsd*col
        return (e - self.origin[1]) / self.gsd, (self.origin[0] - n) / self.gsd

    def ortho_patch(self, frame: np.ndarray, height_agl_m: float, roll: float, pitch: float, yaw: float):
        """North-up patch centred below the camera; returns (patch, fully_inside)."""
        r_nb = body_to_ned(roll, pitch, yaw)
        r_nc = r_nb @ self.r_bc
        js, is_ = np.meshgrid(np.arange(self.npx), np.arange(self.npx))
        dn = (self.npx / 2 - is_ - 0.5) * self.gsd
        de = (js - self.npx / 2 + 0.5) * self.gsd
        # NED z is down: a camera below the body has a smaller AGL.
        camera_height = height_agl_m - (r_nb @ self.offset)[2]
        if camera_height <= 0:
            return None, False
        pts = np.stack([dn, de, np.full_like(dn, camera_height)], -1)
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
        if np.asarray(prior_ne_m).shape != (2,) or radius_m <= 0:
            return MapLocalizationFix(False, "INVALID_SEARCH_WINDOW")
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
        window = self.map_eq[r0:r0 + 2 * rad, c0:c0 + 2 * rad]
        # Search over heading offsets (+-12 deg): with a 4 deg autopilot
        # heading error the correlation lost its margin in SITL; at 6-7 deg the
        # +-6 deg search left the map unmatched after lock. The heading is a
        # search nuisance here, not a measurement (it also absorbed a
        # systematic rotation between the 2019 and 2025 products).
        best = None
        envelope = None
        peaks: dict[float, float] = {}
        for dyaw in self.yaw_search:
            patch, inside = self.ortho_patch(image, height_agl_m, roll_rad, pitch_rad, yaw_rad + dyaw)
            if not inside:
                continue
            patch = self.clahe.apply(patch)
            if float(np.std(patch)) < 5.0:
                continue
            r = cv2.matchTemplate(window, patch, cv2.TM_CCOEFF_NORMED)
            envelope = r.copy() if envelope is None else np.maximum(envelope, r)
            _, pk, _, lc = cv2.minMaxLoc(r)
            peaks[dyaw] = float(pk)
            if best is None or pk > best[0]:
                best = (pk, lc, r, dyaw)
        if best is None:
            return MapLocalizationFix(False, "NO_USABLE_ORTHO_PATCH")
        peak, loc, res, best_dyaw = best
        # Compare all heading hypotheses: a different position at another
        # heading is also a competing map match. The selected surface alone
        # misses such ambiguity.
        masked = envelope.copy()
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
        yaw = self._refined_yaw(yaw_rad, best_dyaw, peaks)
        return MapLocalizationFix(True, "NCC_ACCEPTED_UNVALIDATED", "ncc", float(north), float(east),
                                  0, int(round(100 * peak)), float(peak - second), float("nan"),
                                  float("nan"), False, yaw, float(cam_n), float(cam_e))

    def _refined_yaw(self, yaw_rad: float, best_dyaw: float, peaks: dict[float, float]) -> float:
        """Heading of the best rotation, refined by a parabola through the
        neighbouring rotations. NaN at the edge of the search (the true
        rotation may lie outside it) or without both neighbours. Its error
        statistics need calibration on separate flights, like the position.
        """
        grid = sorted(self.yaw_search)
        i = grid.index(best_dyaw)
        if i == 0 or i == len(grid) - 1 or grid[i - 1] not in peaks or grid[i + 1] not in peaks:
            return float("nan")
        a, b, c = peaks[grid[i - 1]], peaks[best_dyaw], peaks[grid[i + 1]]
        step = grid[i + 1] - best_dyaw
        den = a - 2 * b + c
        d = 0.5 * (a - c) / den * step if den < 0 else 0.0
        y = yaw_rad + best_dyaw + d
        return float(np.arctan2(np.sin(y), np.cos(y)))
