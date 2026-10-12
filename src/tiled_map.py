"""Большой ортофотоплан: перекрывающиеся тайлы и поиск около прогноза.

``OrthophotoLocalizer`` строит фиксированное число признаков ORB на весь
лист. На карте 520×520 м отпечаток камеры (около 54×40 м с 40 м)
получает около 20 признаков — меньше минимума сопоставления. Поэтому
карта режется на тайлы с перекрытием 50 % (каждый отпечаток целиком
лежит хотя бы в одном тайле), и для каждого тайла — свой локализатор.

Запрос ищет только в тайлах, центр которых ближе ``radius`` к прогнозу
положения (положение EKF при ГНСС, затем положение ``MapAnchor``).
Проверка неоднозначности действует внутри области поиска; ее радиус
должен покрывать неопределенность прогноза. Без прогноза поиск идет по
всем тайлам.
"""
from __future__ import annotations

import numpy as np

from .orthophoto_localization import MapGeoReference, MapLocalizationFix, OrthophotoLocalizer, OrthophotoTile


class TiledOrthophotoLocalizer:
    def __init__(self, gray: np.ndarray, georef: MapGeoReference, calibration, mount, *,
                 tile_px: int = 400, stride_px: int = 200, **localizer_kwargs) -> None:
        h, w = gray.shape
        self.entries: list[tuple[np.ndarray, OrthophotoLocalizer]] = []
        jac = georef.jacobian()
        origin = np.asarray(georef.origin_ne_m, dtype=float)
        ys = list(range(0, max(1, h - tile_px) + 1, stride_px))
        xs = list(range(0, max(1, w - tile_px) + 1, stride_px))
        if ys[-1] + tile_px < h:
            ys.append(h - tile_px)
        if xs[-1] + tile_px < w:
            xs.append(w - tile_px)
        for y0 in ys:
            for x0 in xs:
                sub = gray[y0:y0 + tile_px, x0:x0 + tile_px]
                # Same affine georeference shifted to the tile's first pixel.
                sub_origin = origin + jac @ np.array([x0, y0], dtype=float)
                geo = MapGeoReference(tuple(sub_origin), tuple(map(tuple, jac)))
                try:
                    loc = OrthophotoLocalizer([OrthophotoTile(f"{y0}_{x0}", np.ascontiguousarray(sub), geo)],
                                              calibration, mount, **localizer_kwargs)
                except ValueError:
                    continue  # no usable texture (water, uniform grass)
                centre = sub_origin + jac @ np.array([sub.shape[1] / 2, sub.shape[0] / 2])
                self.entries.append((centre, loc))
        if not self.entries:
            raise ValueError("Нет пригодных тайлов ортофотоплана")
        self.last_searched = 0

    def localize(self, image: np.ndarray, *, prior_ne_m=None, radius_m: float = 60.0,
                 **pose) -> MapLocalizationFix:
        if prior_ne_m is None or not np.isfinite(prior_ne_m).all():
            chosen = self.entries
        else:
            p = np.asarray(prior_ne_m, dtype=float)
            chosen = [e for e in self.entries if np.linalg.norm(e[0] - p) <= radius_m]
        self.last_searched = len(chosen)
        if not chosen:
            return MapLocalizationFix(False, "OUTSIDE_MAP")
        best = None
        accepted = []
        for _, loc in chosen:
            fix = loc.localize(image, **pose)
            if fix.accepted:
                accepted.append(fix)
            elif best is None:
                best = fix
        if not accepted:
            return best or MapLocalizationFix(False, "NO_GEOMETRICALLY_VALID_MAP_MATCH")
        accepted.sort(key=lambda f: f.inlier_points, reverse=True)
        top = accepted[0]
        # Overlapping tiles see the same place: only a DISTANT, comparably
        # supported solution is a competing hypothesis.
        for other in accepted[1:]:
            if (np.hypot(other.north_m - top.north_m, other.east_m - top.east_m) > 2.0
                    and other.inlier_points >= 0.75 * top.inlier_points):
                return MapLocalizationFix(False, "AMBIGUOUS_MAP_MATCH")
        return top
