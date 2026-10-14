#!/usr/bin/env python3
"""Ground-only onboard map: mask elevated objects with the swisstopo DSM/DTM.

SWISSIMAGE is not a true orthophoto: roofs and tree crowns are displaced
from their ground position by h * tan(view angle) of the aerial image, and
differently in each year's product. Map fixes over such places were offset
by 1-2 m, a few places by 4-21 m (series H). A true orthophoto needs the
original images; here the elevated objects are removed instead:

  nDSM = swissSURFACE3D (DSM, 2018) - swissALTI3D (DTM, 2020), 0.5 m;
  elevated = nDSM > h_min, dilated by k * h (k = 0.25: chosen among 0,
  0.25 and 0.5 on calibration frames only); masked pixels are replaced by a strongly blurred map,
  so they carry no edges for the correlation.

The camera still sees roofs; only the map side is ground-only. Not a true
orthophoto: a ground-only approximation.

Tiles (GeoTIFF, LV95, 0.5 m, pixel-is-area):
  https://data.geo.admin.ch/ch.swisstopo.swisssurface3d-raster/swisssurface3d-raster_2018_2683-1250/
      swisssurface3d-raster_2018_2683-1250_0.5_2056_5728.tif
  https://data.geo.admin.ch/ch.swisstopo.swissalti3d/swissalti3d_2020_2683-1250/
      swissalti3d_2020_2683-1250_0.5_2056_5728.tif
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from swiss_scene import WORLD_LAT, WORLD_LON, wgs84_to_lv95  # noqa: E402

DSM_TILE_EAST_M, DSM_TILE_NORTH_TOP_M, DSM_GSD_M = 2683000.0, 1251000.0, 0.5


def object_heights_on_map(dsm: np.ndarray, dtm: np.ndarray, georef: dict, shape: tuple[int, int]) -> np.ndarray:
    """nDSM resampled on the map grid (bilinear), metres above ground."""
    e0, n0 = wgs84_to_lv95(WORLD_LAT, WORLD_LON)
    (on, oe) = georef["origin_ne_m"]
    gsd = georef["matrix_ne_m_per_px"][1][0]
    rows, cols = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float32)
    north = n0 + on - gsd * rows          # pixel centres, local N/E -> LV95
    east = e0 + oe + gsd * cols
    # DSM pixel (0, 0) covers [E0, E0 + 0.5] x [N0 - 0.5, N0]; its centre is at +0.25.
    mx = ((east - DSM_TILE_EAST_M) / DSM_GSD_M - 0.5).astype(np.float32)
    my = ((DSM_TILE_NORTH_TOP_M - north) / DSM_GSD_M - 0.5).astype(np.float32)
    ndsm = (dsm - dtm).astype(np.float32)
    return cv2.remap(ndsm, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)


def elevated_mask(heights: np.ndarray, gsd: float, *, h_min: float = 2.0, k: float = 0.5) -> np.ndarray:
    """Elevated objects dilated by k * height (per height class)."""
    mask = np.zeros(heights.shape, bool)
    for h in (h_min, 5.0, 10.0, 20.0, 40.0):
        sel = heights > h
        if not sel.any():
            continue
        r = max(1, int(round(k * max(h, h_min) / gsd)))
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        mask |= cv2.dilate(sel.astype(np.uint8), kern) > 0
    return mask


def ground_map(gray: np.ndarray, mask: np.ndarray, gsd: float, blur_m: float = 8.0) -> np.ndarray:
    blurred = cv2.GaussianBlur(gray, (0, 0), blur_m / gsd)
    out = gray.copy()
    out[mask] = blurred[mask]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("scene_dir", type=Path, help="output of swiss_scene.py (orthophoto.png, orthophoto_georef.json)")
    ap.add_argument("--dsm", type=Path, required=True)
    ap.add_argument("--dtm", type=Path, required=True)
    ap.add_argument("--h-min", type=float, default=2.0)
    ap.add_argument("--k", type=float, default=0.25,
                    help="dilation per metre of object height (0.25 chosen on the series H calibration frames)")
    args = ap.parse_args()
    import tifffile
    georef = json.loads((args.scene_dir / "orthophoto_georef.json").read_text())
    gray = cv2.imread(str(args.scene_dir / "orthophoto.png"), cv2.IMREAD_GRAYSCALE)
    heights = object_heights_on_map(tifffile.imread(args.dsm), tifffile.imread(args.dtm), georef, gray.shape)
    gsd = georef["matrix_ne_m_per_px"][1][0]
    mask = elevated_mask(heights, gsd, h_min=args.h_min, k=args.k)
    cv2.imwrite(str(args.scene_dir / "orthophoto_ground.png"), ground_map(gray, mask, gsd))
    cv2.imwrite(str(args.scene_dir / "orthophoto_ground_mask.png"), mask.astype(np.uint8) * 255)
    info = {"masked_fraction": float(mask.mean()), "h_min_m": args.h_min, "dilation_k": args.k,
            "max_object_height_m": float(heights.max()), "dsm": args.dsm.name, "dtm": args.dtm.name}
    (args.scene_dir / "orthophoto_ground.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info))


if __name__ == "__main__":
    main()
