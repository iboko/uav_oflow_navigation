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
                 map_clahe_tile_m: float = 8.75, shape_radius_m: float = 1.0,
                 subpatch_search_m: float = 3.0, surface_m: np.ndarray | None = None,
                 occlusion_tolerance_m: float = 0.5, max_outside_fraction: float = 0.1) -> None:
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
        self.shape_radius_px = max(1, int(round(shape_radius_m / self.gsd)))
        self.subpatch_search_px = max(1, int(round(subpatch_search_m / self.gsd)))
        # Surface model (DSM) on the map grid [m, up, any common datum]: the
        # frame is projected onto it instead of a flat ground.
        if surface_m is not None and np.asarray(surface_m).shape != gray.shape:
            raise ValueError("Модель поверхности должна совпадать с картой по размеру")
        self.surface = None if surface_m is None else np.asarray(surface_m, dtype=np.float32)
        self.occlusion_tolerance_m = occlusion_tolerance_m
        self.max_outside_fraction = max_outside_fraction
        self.last_hidden_fraction = 0.0
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

    def ortho_patch(self, frame: np.ndarray, height_agl_m: float, roll: float, pitch: float, yaw: float,
                    center_ne_m=None):
        """North-up patch centred below the camera; returns (patch, fully_inside).

        Without a surface model: every patch cell is a point of a flat ground
        ``height_agl_m`` below the camera. With one (``surface_m``, heights on
        the map grid) and the camera's ground position ``center_ne_m``: the
        cell is the surface point (N, E, z(N, E)); the camera is
        ``height_agl_m`` (rangefinder: distance to the surface below) above
        z at its nadir. Cells hidden from the camera by a nearer surface
        (z-buffer in the image) take the mean of the visible cells.
        """
        r_nb = body_to_ned(roll, pitch, yaw)
        r_nc = r_nb @ self.r_bc
        js, is_ = np.meshgrid(np.arange(self.npx), np.arange(self.npx))
        dn = (self.npx / 2 - is_ - 0.5) * self.gsd
        de = (js - self.npx / 2 + 0.5) * self.gsd
        # NED z is down: a camera below the body has a smaller AGL.
        camera_height = height_agl_m - (r_nb @ self.offset)[2]
        if camera_height <= 0:
            return None, False
        depth = np.full_like(dn, camera_height)
        use_surface = self.surface is not None and center_ne_m is not None
        if use_surface:
            cn, ce = float(center_ne_m[0]), float(center_ne_m[1])
            z_cells = self._surface_at(cn + dn, ce + de)
            z_nadir = float(self._surface_at(np.array([[cn]]), np.array([[ce]]))[0, 0])
            depth = camera_height + z_nadir - z_cells
        pts = np.stack([dn, de, depth], -1)
        pc = pts @ r_nc
        uv = pc @ self.k.T
        mx = (uv[..., 0] / uv[..., 2]).astype(np.float32)
        my = (uv[..., 1] / uv[..., 2]).astype(np.float32)
        h, w = frame.shape
        in_frame = (mx >= 0) & (mx < w - 1) & (my >= 0) & (my < h - 1) & (pc[..., 2] > 0)
        # With relief, roof cells come closer to the camera and can leave the
        # frame: up to max_outside_fraction of the cells are allowed out
        # (filled like hidden cells); a flat ground needs the whole patch.
        inside = bool(in_frame.all()) if not use_surface else float(in_frame.mean()) >= 1 - self.max_outside_fraction
        patch = cv2.remap(frame, mx, my, cv2.INTER_LINEAR)
        if use_surface and inside:
            # Occlusion: the nearest cell wins each image pixel.
            px = np.clip(np.round(mx).astype(int), 0, w - 1)
            py = np.clip(np.round(my).astype(int), 0, h - 1)
            zbuf = np.full((h, w), np.inf)
            sel = in_frame.ravel()
            np.minimum.at(zbuf, (py.ravel()[sel], px.ravel()[sel]), pc[..., 2].ravel()[sel])
            hidden = (pc[..., 2] > zbuf[py, px] + self.occlusion_tolerance_m) | ~in_frame
            if hidden.any() and not hidden.all():
                patch = patch.copy()
                patch[hidden] = np.uint8(round(float(patch[~hidden].mean())))
            self.last_hidden_fraction = float(hidden.mean())
        return patch, inside

    def _surface_at(self, north: np.ndarray, east: np.ndarray) -> np.ndarray:
        col = ((east - self.origin[1]) / self.gsd).astype(np.float32)
        row = ((self.origin[0] - north) / self.gsd).astype(np.float32)
        return cv2.remap(self.surface, col, row, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

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
        locs: dict[float, tuple[int, int]] = {}
        for dyaw in self.yaw_search:
            patch, inside = self.ortho_patch(image, height_agl_m, roll_rad, pitch_rad, yaw_rad + dyaw,
                                             center_ne_m=cam_prior)
            if not inside:
                continue
            patch = self.clahe.apply(patch)
            if float(np.std(patch)) < 5.0:
                continue
            r = cv2.matchTemplate(window, patch, cv2.TM_CCOEFF_NORMED)
            envelope = r.copy() if envelope is None else np.maximum(envelope, r)
            _, pk, _, lc = cv2.minMaxLoc(r)
            peaks[dyaw] = float(pk)
            locs[dyaw] = lc
            if best is None or pk > best[0]:
                best = (pk, lc, r, dyaw)
                best_patch = patch
        if best is None:
            return MapLocalizationFix(False, "NO_USABLE_ORTHO_PATCH")
        peak, loc, res, best_dyaw = best
        self.last_locs, self.last_peaks = locs, peaks
        self.last_window, self.last_patch, self.last_loc = window, best_patch, loc
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
                                  float("nan"), False, yaw, float(cam_n), float(cam_e),
                                  self._peak_shape(res, loc, float(peak)),
                                  self._subpatch_cov(window, best_patch, loc))

    def _subpatch_cov(self, window, patch, loc):
        sub = self.subpatch_offsets(window, patch, loc, self.subpatch_search_px)
        if sub is None:
            return None
        c = np.cov(sub.T, ddof=1)
        return ((float(c[0, 0]), float(c[0, 1])), (float(c[1, 0]), float(c[1, 1])))

    def subpatch_offsets(self, window: np.ndarray, patch: np.ndarray, loc, search_px: int) -> np.ndarray | None:
        """N/E offsets [m] of the 4 patch quadrants matched separately within
        +-search_px of the full-patch solution (same heading).

        Relief displacement is not the same in two orthophotos (roofs lean
        differently in each year's non-true orthophoto): the quadrants then
        disagree. Their scatter measures this local geometric mismatch,
        which the full-patch peak shape does not see.
        """
        h = self.npx // 2
        out = []
        for qy in (0, h):
            for qx in (0, h):
                q = patch[qy:qy + h, qx:qx + h]
                if float(np.std(q)) < 5.0:
                    continue
                y0, x0 = loc[1] + qy - search_px, loc[0] + qx - search_px
                if y0 < 0 or x0 < 0 or y0 + h + 2 * search_px > window.shape[0] \
                        or x0 + h + 2 * search_px > window.shape[1]:
                    continue
                r = cv2.matchTemplate(window[y0:y0 + h + 2 * search_px, x0:x0 + h + 2 * search_px], q,
                                      cv2.TM_CCOEFF_NORMED)
                _, _, _, lc = cv2.minMaxLoc(r)
                dc, dr = lc[0] - search_px, lc[1] - search_px
                out.append((-dr * self.gsd, dc * self.gsd))
        return np.asarray(out) if len(out) >= 3 else None

    def _peak_shape(self, res: np.ndarray, loc, peak: float):
        """(1 - peak) * (-H)^-1 in N/E [m^2]; None if the peak is not a
        well-formed maximum.

        H: least-squares quadratic over a (2r+1)^2 neighbourhood of the
        integer peak (r = shape_radius_px). A flat direction of the
        correlation surface (a ridge, e.g. a straight edge) gives a large
        shape along it: that is where the match is weak. Its scale relative
        to the actual error is calibrated, never assumed.
        """
        r = self.shape_radius_px
        cx, cy = loc
        if not (r <= cx < res.shape[1] - r and r <= cy < res.shape[0] - r):
            return None
        ys, xs = np.mgrid[-r:r + 1, -r:r + 1]
        x, y = xs.ravel().astype(float), ys.ravel().astype(float)
        z = res[cy - r:cy + r + 1, cx - r:cx + r + 1].ravel().astype(float)
        a = np.column_stack([np.ones_like(x), x, y, 0.5 * x * x, x * y, 0.5 * y * y])
        coef = np.linalg.lstsq(a, z, rcond=None)[0]
        h = np.array([[coef[3], coef[4]], [coef[4], coef[5]]])  # per px^2, (col, row)
        w = np.linalg.eigvalsh(h)
        if not np.all(np.isfinite(w)) or w.max() >= 0:
            return None
        cov_px = max(1.0 - peak, 1e-3) * np.linalg.inv(-h)
        # (col, row) -> (N, E): dN = -gsd * drow, dE = gsd * dcol.
        j = np.array([[0.0, -self.gsd], [self.gsd, 0.0]])
        c = j @ cov_px @ j.T
        return ((float(c[0, 0]), float(c[0, 1])), (float(c[1, 0]), float(c[1, 1])))

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
