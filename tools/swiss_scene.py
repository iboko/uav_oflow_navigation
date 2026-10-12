#!/usr/bin/env python3
"""Gazebo scene and onboard orthophoto from real swisstopo SWISSIMAGE tiles.

The PX4 Gazebo worlds sit at 47.39797 N, 8.54616 E (Zurich). SWISSIMAGE
10 cm (open data, swisstopo) covers it in several years. The scene ground
is the 2025 orthophoto, the onboard map is the 2019 one: different date,
season, shadows, cars, buildings - a real map-matching problem, unlike a
degraded copy of the scene texture.

Both are cropped on the LV95 grid around the world origin; the world axes
are the LV95 grid axes (east, north). The map georeference is the exact
LV95 one: whatever offset exists between the 2019 and 2025 products is a
real survey difference, nothing is injected.

Tiles (GeoTIFF, LV95, 0.1 m):
  https://data.geo.admin.ch/ch.swisstopo.swissimage-dop10/swissimage-dop10_<YEAR>_2683-1250/
      swissimage-dop10_<YEAR>_2683-1250_0.1_2056.tif
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

WORLD = "ofnav_swiss"
WORLD_LAT, WORLD_LON = 47.397971057728974, 8.546163739800146  # PX4 default.sdf
TILE_EAST_M, TILE_NORTH_TOP_M, GSD_M = 2683000.0, 1251000.0, 0.1


def wgs84_to_lv95(lat: float, lon: float) -> tuple[float, float]:
    """swisstopo approximate formulas (about 1 m)."""
    p = (lat * 3600 - 169028.66) / 10000
    lam = (lon * 3600 - 26782.5) / 10000
    e = (2600072.37 + 211455.93 * lam - 10938.51 * lam * p - 0.36 * lam * p * p - 44.54 * lam ** 3)
    n = (1200147.07 + 308807.95 * p + 3745.25 * lam * lam + 76.63 * p * p - 194.56 * lam * lam * p
         + 119.79 * p ** 3)
    return e, n


def crop(tile: Path, center_en: tuple[float, float], size_m: float):
    """Crop (RGB) centred on center_en; returns image and LV95 of its top-left corner."""
    import tifffile
    img = tifffile.imread(str(tile))
    half = int(round(size_m / GSD_M / 2))
    c = int(round((center_en[0] - TILE_EAST_M) / GSD_M))
    r = int(round((TILE_NORTH_TOP_M - center_en[1]) / GSD_M))
    if r - half < 0 or c - half < 0 or r + half > img.shape[0] or c + half > img.shape[1]:
        raise ValueError("Сцена выходит за пределы тайла")
    out = img[r - half:r + half, c - half:c + half]
    return out, (TILE_EAST_M + (c - half) * GSD_M, TILE_NORTH_TOP_M - (r - half) * GSD_M)


def build(px4_dir: Path, scene_tile: Path, map_tile: Path, out: Path, *, size_m: float = 400.0,
          map_gsd_m: float = 0.25) -> dict:
    import cv2
    out.mkdir(parents=True, exist_ok=True)
    origin = wgs84_to_lv95(WORLD_LAT, WORLD_LON)
    worlds = px4_dir / "Tools" / "simulation" / "gz" / "worlds"
    texture, tl = crop(scene_tile, origin, size_m)
    tex_path = worlds / f"{WORLD}.png"
    cv2.imwrite(str(tex_path), cv2.cvtColor(texture, cv2.COLOR_RGB2BGR))
    # Plane centre in world ENU (the crop is aligned to whole pixels).
    cx = tl[0] + size_m / 2 - origin[0]
    cy = tl[1] - size_m / 2 - origin[1]
    sdf = (worlds / "default.sdf").read_text(encoding="utf-8-sig")
    sdf = sdf.replace('<world name="default">', f'<world name="{WORLD}">', 1)
    stock = sdf[sdf.index("<visual name=\"visual\">"):]
    stock = stock[:stock.index("</visual>") + len("</visual>")]
    sdf = sdf.replace(stock, f"""<visual name="visual">
          <pose>{cx:.3f} {cy:.3f} 0 0 0 0</pose>
          <geometry><plane><normal>0 0 1</normal><size>{size_m} {size_m}</size></plane></geometry>
          <material>
            <ambient>1 1 1 1</ambient><diffuse>1 1 1 1</diffuse><specular>0 0 0 1</specular>
            <pbr><metal><albedo_map>{tex_path}</albedo_map><roughness>1</roughness></metal></pbr>
          </material>
        </visual>""", 1)
    (worlds / f"{WORLD}.sdf").write_text(sdf, encoding="utf-8")
    # Onboard map: another year, coarser, generous margin around the scene.
    margin = 60.0
    m_img, m_tl = crop(map_tile, origin, size_m + 2 * margin)
    scale = GSD_M / map_gsd_m
    m_img = cv2.resize(m_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(m_img, cv2.COLOR_RGB2GRAY)
    map_png = out / "orthophoto.png"
    cv2.imwrite(str(map_png), gray)
    georef = {"origin_ne_m": [m_tl[1] - origin[1] - 0.5 * map_gsd_m, m_tl[0] - origin[0] + 0.5 * map_gsd_m],
              "matrix_ne_m_per_px": [[0.0, -map_gsd_m], [map_gsd_m, 0.0]],
              "note": "SWISSIMAGE map year on LV95; world origin at LV95 "
                      f"E {origin[0]:.2f} N {origin[1]:.2f}; no injected error"}
    (out / "orthophoto_georef.json").write_text(json.dumps(georef, indent=2))
    info = {"world": WORLD, "origin_lv95": origin, "scene_top_left_lv95": tl, "plane_center_enu": [cx, cy],
            "size_m": size_m, "scene_tile": scene_tile.name, "map_tile": map_tile.name,
            "map_gsd_m": map_gsd_m, "map_png": str(map_png)}
    (out / "scene.json").write_text(json.dumps(info, indent=2))
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("px4_dir", type=Path)
    ap.add_argument("--scene-tile", type=Path, required=True, help="SWISSIMAGE GeoTIFF for the ground (e.g. 2025)")
    ap.add_argument("--map-tile", type=Path, required=True, help="SWISSIMAGE GeoTIFF for the onboard map (e.g. 2019)")
    ap.add_argument("--out", type=Path, default=Path("build/swiss_scene"))
    ap.add_argument("--size-m", type=float, default=400.0)
    args = ap.parse_args()
    print(json.dumps(build(args.px4_dir.resolve(), args.scene_tile, args.map_tile, args.out, size_m=args.size_m),
                     indent=2))


if __name__ == "__main__":
    main()
