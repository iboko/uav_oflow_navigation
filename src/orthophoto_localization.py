"""Conservative map-aided localization of downward UAV camera on an orthophoto.

Offline experiment only. Orthophoto pixel centres have explicit affine N/E
coordinates (metres in a LOCAL NED-aligned plane); not raw lat/lon degrees.
The output is a candidate global horizontal position, NOT an EKF update.
No covariance is invented from RANSAC residuals. Camera frames must be
undistorted and map imagery rectified to the same terrain plane/datum.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import cv2
import numpy as np

from .camera_rotation_compensation import CameraMountCalibration, body_to_ned
from .ground_visual_motion import CameraCalibration


def _numeric_matrix(value: object, size: tuple[int, ...], name: str) -> np.ndarray:
    try:
        a = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Некорректные числовые значения: {name}") from exc
    if a.shape != size or not np.isfinite(a).all():
        raise ValueError(f"Неверная размерность или неконечные данные: {name}")
    return a


@dataclass(frozen=True)
class MapGeoReference:
    """Map pixel -> local (N,E) coordinates in metres.

    ne = origin_ne_m + matrix_ne_m_per_px @ [u,v].
    Affine grid is assumed already georeferenced and orthorectified; if
    original CRS is geographic degrees it must be projected beforehand.
    """

    origin_ne_m: tuple[float, float]
    matrix_ne_m_per_px: tuple[tuple[float, float], tuple[float, float]]

    def transform(self, pixel: np.ndarray) -> np.ndarray:
        origin = _numeric_matrix(self.origin_ne_m, (2,), "начало карты")
        matrix = _numeric_matrix(self.matrix_ne_m_per_px, (2, 2), "матрица карты")
        singular = np.linalg.svd(matrix, compute_uv=False)
        if singular[0] > 10. or singular[-1] < 0.005 or singular[0]/singular[-1] > 10.:
            raise ValueError("Некорректное пространственное разрешение/деформация карты")
        p = _numeric_matrix(pixel, (2,), "пиксель карты")
        return origin + matrix @ p

    def jacobian(self) -> np.ndarray:
        self.transform(np.zeros(2))
        return np.asarray(self.matrix_ne_m_per_px, dtype=float)


@dataclass(frozen=True)
class OrthophotoTile:
    name: str
    gray: np.ndarray
    georef: MapGeoReference


@dataclass(frozen=True)
class MapLocalizationFix:
    accepted: bool
    status: str
    tile_name: str = ""
    north_m: float = float("nan")
    east_m: float = float("nan")
    matched_points: int = 0
    inlier_points: int = 0
    inlier_fraction: float = 0.0
    median_residual_map_px: float = float("nan")
    geometric_jacobian_error: float = float("nan")
    # No covariance provided: parameters require independent calibration.
    covariance_validated: bool = False


@dataclass(frozen=True)
class _Candidate:
    tile: OrthophotoTile
    ne: np.ndarray
    inliers: int
    matched: int
    ratio: float
    residual: float
    jac_error: float


def _ground_ray_offset(
    u: float, v: float, height_agl_m: float, camera: CameraCalibration,
    r_nc: np.ndarray,
) -> np.ndarray:
    """N/E offset from camera optical centre to ray/ground-plane intersection."""
    ray_cam = np.array([
        (u-camera.center_x_px)/camera.focal_x_px,
        (v-camera.center_y_px)/camera.focal_y_px,
        1.,
    ])
    ray = r_nc @ ray_cam
    if not np.isfinite(ray).all() or ray[2] < 0.6:
        raise ValueError("Луч камеры не направлен на допустимую поверхность")
    return height_agl_m * ray[:2] / ray[2]


def _jacobian_homography(h: np.ndarray, u: float, v: float) -> np.ndarray:
    def transform(pixel):
        pts = np.asarray(pixel, dtype=float).reshape(1, 1, 2)
        return cv2.perspectiveTransform(pts, h).reshape(2)
    p = transform([u, v])
    # symmetric central finite difference avoids normalisation path errors.
    du = (transform([u + .5, v]) - transform([u - .5, v]))
    dv = (transform([u, v + .5]) - transform([u, v - .5]))
    if not np.isfinite(p).all() or not np.isfinite(du).all() or not np.isfinite(dv).all():
        raise ValueError("Неустойчивое проективное преобразование")
    return np.column_stack((du, dv))


class OrthophotoLocalizer:
    """ORB + ratio test + RANSAC + calibrated attitude/scale/ambiguity gates."""

    def __init__(
        self,
        tiles: list[OrthophotoTile],
        calibration: CameraCalibration,
        mount: CameraMountCalibration,
        *,
        camera_offset_body_m: tuple[float, float, float] = (0., 0., 0.),
        max_tilt_rad: float = 0.09,
        min_inliers: int = 25,
        min_inlier_ratio: float = 0.45,
        min_spatial_cells: int = 6,
        max_jacobian_error_fraction: float = 0.30,
        competing_score_fraction: float = 0.75,
    ) -> None:
        if not calibration.valid() or not mount.valid(calibration):
            raise ValueError("Некорректная/несогласованная пространственная калибровка камеры")
        if not 0 < max_tilt_rad <= 0.2 or min_inliers < 12:
            raise ValueError("Недопустимые углы/минимальное число характерных точек")
        if not 0 < min_inlier_ratio < 1 or not 4 <= min_spatial_cells <= 16:
            raise ValueError("Недопустимая проверка сопоставлений")
        if not 0 < max_jacobian_error_fraction <= .5:
            raise ValueError("Недопустимая точность геометрической проверки")
        if not 0 < competing_score_fraction < 1:
            raise ValueError("Недопустимый критерий альтернативной привязки")
        if not tiles or len({t.name for t in tiles}) != len(tiles):
            raise ValueError("Нужны разные идентификаторы листов ортофотоплана")
        self.calibration = calibration
        self.mount = mount
        self.camera_offset_body_m = _numeric_matrix(
            camera_offset_body_m, (3,), "плечо установки камеры"
        )
        if np.linalg.norm(self.camera_offset_body_m) > 5.:
            raise ValueError("Плечо установки превышает диапазон")
        self.max_tilt_rad = max_tilt_rad
        self.min_inliers = min_inliers
        self.min_ratio = min_inlier_ratio
        self.min_cells = min_spatial_cells
        self.max_geom_error = max_jacobian_error_fraction
        self.competing_fraction = competing_score_fraction
        self.tiles = []
        self.orb = cv2.ORB_create(nfeatures=2400, fastThreshold=9)
        for tile in tiles:
            if not tile.name.strip() or not isinstance(tile.gray, np.ndarray):
                raise ValueError("Недопустимое имя/изображение листа")
            if tile.gray.ndim != 2 or tile.gray.dtype != np.uint8 or min(tile.gray.shape) < 200:
                raise ValueError("Нужен ортоснимок в uint8 достаточного размера")
            tile.georef.jacobian()
            keypoints, descriptors = self.orb.detectAndCompute(tile.gray, None)
            if descriptors is not None and len(keypoints) >= min_inliers:
                self.tiles.append((tile, keypoints, descriptors))
        if not self.tiles:
            raise ValueError("Нет пригодных характерных точек на ортофотоплане")

    def localize(
        self,
        image: np.ndarray,
        *,
        height_agl_m: float,
        roll_rad: float,
        pitch_rad: float,
        yaw_rad: float,
    ) -> MapLocalizationFix:
        def reject(reason: str) -> MapLocalizationFix:
            return MapLocalizationFix(False, reason)

        if not isinstance(image, np.ndarray) or image.ndim != 2 or image.dtype != np.uint8:
            return reject("INVALID_IMAGE")
        if min(image.shape) < 120:
            return reject("INVALID_IMAGE")
        if not all(isfinite(x) for x in (height_agl_m, roll_rad, pitch_rad, yaw_rad)):
            return reject("NONFINITE_INPUT")
        if not 2. <= height_agl_m <= 150.:
            return reject("UNSUPPORTED_HEIGHT")
        if abs(roll_rad) > self.max_tilt_rad or abs(pitch_rad) > self.max_tilt_rad:
            return reject("UNSUPPORTED_ATTITUDE")
        mount = np.asarray(self.mount.body_from_camera, dtype=float)
        r_nb = body_to_ned(roll_rad, pitch_rad, yaw_rad)
        r_nc = r_nb @ mount
        principal = np.array([
            self.calibration.center_x_px, self.calibration.center_y_px
        ], dtype=float)
        lever_ne = (r_nb @ self.camera_offset_body_m)[:2]
        try:
            ground_offset = _ground_ray_offset(
                *principal, height_agl_m, self.calibration, r_nc
            )
            expected_jac = np.column_stack((
                _ground_ray_offset(principal[0]+1, principal[1], height_agl_m,
                                   self.calibration, r_nc) - ground_offset,
                _ground_ray_offset(principal[0], principal[1]+1, height_agl_m,
                                   self.calibration, r_nc) - ground_offset,
            ))
        except ValueError:
            return reject("INVALID_CAMERA_RAYS")

        kp_query, des_query = self.orb.detectAndCompute(image, None)
        if des_query is None or len(kp_query) < self.min_inliers:
            return reject("INSUFFICIENT_TEXTURE")
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        candidates: list[_Candidate] = []
        for tile, kp_map, des_map in self.tiles:
            pairs = bf.knnMatch(des_query, des_map, k=2)
            good = [a for a, b in pairs if a.distance < 0.72*b.distance]
            if len(good) < self.min_inliers:
                continue
            src = np.asarray([kp_query[m.queryIdx].pt for m in good], dtype=np.float32)
            dst = np.asarray([kp_map[m.trainIdx].pt for m in good], dtype=np.float32)
            h, mask = cv2.findHomography(
                src, dst, cv2.RANSAC, 2.5, maxIters=3500, confidence=0.997
            )
            if h is None or mask is None or not np.isfinite(h).all():
                continue
            inlier_mask = mask.ravel().astype(bool)
            inlier_count = int(np.sum(inlier_mask))
            fraction = inlier_count / len(good)
            if inlier_count < self.min_inliers or fraction < self.min_ratio:
                continue
            query_inliers = src[inlier_mask]
            cols = np.clip((4 * query_inliers[:, 0] / image.shape[1]).astype(int),0,3)
            rows = np.clip((4 * query_inliers[:, 1] / image.shape[0]).astype(int),0,3)
            if len(set(zip(cols.tolist(), rows.tolist()))) < self.min_cells:
                continue
            try:
                mapped = cv2.perspectiveTransform(
                    query_inliers.reshape(-1,1,2), h
                ).reshape(-1,2)
                residual = np.linalg.norm(mapped - dst[inlier_mask], axis=1)
                median = float(np.median(residual))
                if median > 2.0 or not np.isfinite(median):
                    continue
                mapped_center = cv2.perspectiveTransform(
                    principal.astype(np.float32).reshape(1,1,2), h
                ).reshape(2)
                if not (-5 <= mapped_center[0] < tile.gray.shape[1]+5 and
                        -5 <= mapped_center[1] < tile.gray.shape[0]+5):
                    continue
                observed_jac = tile.georef.jacobian() @ _jacobian_homography(
                    h, *principal
                )
                jac_error = float(np.linalg.norm(observed_jac-expected_jac) /
                                  np.linalg.norm(expected_jac))
                if not np.isfinite(jac_error) or jac_error > self.max_geom_error:
                    continue
                ground = tile.georef.transform(mapped_center)
                ne_vehicle = ground - ground_offset - lever_ne
                if not np.isfinite(ne_vehicle).all():
                    continue
                candidates.append(_Candidate(
                    tile, ne_vehicle, inlier_count, len(good),
                    fraction, median, jac_error
                ))
            except (ValueError, cv2.error, np.linalg.LinAlgError):
                continue
        if not candidates:
            return reject("NO_GEOMETRICALLY_VALID_MAP_MATCH")
        candidates.sort(key=lambda c: (c.inliers, c.ratio, -c.residual), reverse=True)
        best = candidates[0]
        if len(candidates) > 1:
            other = candidates[1]
            # No claim of uniqueness when a second sheet explains the same
            # view nearly as well. Spatially overlapping identical tiles may
            # also be ambiguous; safest behaviour is to reject.
            if other.inliers >= self.competing_fraction * best.inliers:
                return reject("AMBIGUOUS_MAP_MATCH")
        return MapLocalizationFix(
            True, "GEOMETRICALLY_ACCEPTED_UNVALIDATED", best.tile.name,
            float(best.ne[0]), float(best.ne[1]),
            best.matched, best.inliers, best.ratio,
            best.residual, best.jac_error, False,
        )
