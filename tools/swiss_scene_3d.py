#!/usr/bin/env python3
"""3D Gazebo scene: the swisstopo surface model (DSM) textured with SWISSIMAGE.

The flat scene (``swiss_scene.py``) is a plane with the 2025 orthophoto on
it: the camera sees no parallax of buildings and the rangefinder always
measures the ground. Projecting a frame onto a surface model can only be
tested where the world has relief. Here the ground is a mesh of
swissSURFACE3D Raster (DSM, 2018, 0.5 m) sampled every ``step_m``, heights
relative to the ground at the world origin. Collision: the terrain without
buildings (DTM) every 5 m - a collision mesh of the full DSM broke the
physics; buildings are not obstacles, the flights stay above them.

Limitation: the texture is the 2025 orthophoto, which is not a true
orthophoto: roof pixels in it are displaced from the roofs of the mesh by
the lean of the 2025 aerial images (walls and roof edges get wrong
colours). The DSM is from 2018: buildings changed since then do not match
the texture either.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from swiss_scene import WORLD_LAT, WORLD_LON, wgs84_to_lv95

WORLD_3D = "ofnav_swiss3d"
DSM_E0, DSM_N0, DSM_GSD = 2683000.0, 1251000.0, 0.5


def sample(raster: np.ndarray, east: np.ndarray, north: np.ndarray) -> np.ndarray:
    """Bilinear sample of a 0.5 m LV95 tile (pixel-is-area) at LV95 points."""
    mx = ((east - DSM_E0) / DSM_GSD - 0.5).astype(np.float32)
    my = ((DSM_N0 - north) / DSM_GSD - 0.5).astype(np.float32)
    return cv2.remap(raster.astype(np.float32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def write_obj(path: Path, x: np.ndarray, y: np.ndarray, z: np.ndarray, u: np.ndarray, v: np.ndarray,
              texture: str) -> None:
    n_r, n_c = x.shape
    mtl = path.with_suffix(".mtl")
    mtl.write_text(f"newmtl ground\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\nmap_Kd {texture}\n")
    idx = np.arange(n_r * n_c).reshape(n_r, n_c) + 1
    a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
    # Rows run south -> north, columns west -> east: counter-clockwise from above.
    tris = np.concatenate([np.stack([a, b, d], -1).reshape(-1, 3), np.stack([a, d, c], -1).reshape(-1, 3)])
    with path.open("w") as f:
        f.write(f"mtllib {mtl.name}\nusemtl ground\n")
        f.write("".join(f"v {p:.3f} {q:.3f} {r:.3f}\n" for p, q, r in zip(x.ravel(), y.ravel(), z.ravel())))
        f.write("".join(f"vt {p:.6f} {q:.6f}\n" for p, q in zip(u.ravel(), v.ravel())))
        # Vertex normals from the height gradient (needed by the mesh loaders).
        gy, gx = np.gradient(z, y[:, 0], x[0, :])
        nrm = np.stack([-gx, -gy, np.ones_like(z)], -1)
        nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
        f.write("".join(f"vn {a:.4f} {b:.4f} {c:.4f}\n" for a, b, c in nrm.reshape(-1, 3)))
        f.write("".join(f"f {i}/{i}/{i} {j}/{j}/{j} {k}/{k}/{k}\n" for i, j, k in tris))


def build(px4_dir: Path, dsm_path: Path, dtm_path: Path, scene_json: Path, *, step_m: float = 1.0,
          collision_step_m: float = 5.0) -> dict:
    import tifffile
    scene = json.loads(scene_json.read_text())
    size = float(scene["size_m"])
    tl_e, tl_n = scene["scene_top_left_lv95"]
    e0, n0 = wgs84_to_lv95(WORLD_LAT, WORLD_LON)
    dsm, dtm = tifffile.imread(dsm_path), tifffile.imread(dtm_path)
    ground0 = float(sample(dtm, np.array([[e0]]), np.array([[n0]]))[0, 0])
    k = int(round(size / step_m)) + 1
    east = tl_e + np.arange(k) * step_m                    # west -> east
    north = (tl_n - size) + np.arange(k) * step_m          # south -> north
    ee, nn = np.meshgrid(east, north)
    z = sample(dsm, ee, nn) - ground0
    worlds = px4_dir / "Tools" / "simulation" / "gz" / "worlds"
    obj = worlds / f"{WORLD_3D}.obj"
    u = (ee - tl_e) / size
    v = (nn - (tl_n - size)) / size                         # OBJ v grows upwards (north)
    write_obj(obj, ee - e0, nn - n0, z, u, v, f"{scene['world']}.png")
    # Collision: the terrain without buildings (DTM), coarse enough for the
    # physics engine; landings anywhere on the route end on the ground.
    kc = int(round(size / collision_step_m)) + 1
    ec = tl_e + np.arange(kc) * collision_step_m
    nc = (tl_n - size) + np.arange(kc) * collision_step_m
    eec, nnc = np.meshgrid(ec, nc)
    zc = sample(dtm, eec, nnc) - ground0
    obj_c = worlds / f"{WORLD_3D}_collision.obj"
    write_obj(obj_c, eec - e0, nnc - n0, zc, np.zeros_like(zc), np.zeros_like(zc), f"{scene['world']}.png")
    sdf = (worlds / "default.sdf").read_text(encoding="utf-8-sig")
    sdf = sdf.replace('<world name="default">', f'<world name="{WORLD_3D}">', 1)
    start = sdf.index('<model name="ground_plane">')
    end = sdf.index("</model>", start) + len("</model>")
    sdf = sdf[:start] + f"""<model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision"><geometry><mesh><uri>{obj_c}</uri></mesh></geometry></collision>
        <visual name="visual"><geometry><mesh><uri>{obj}</uri></mesh></geometry></visual>
      </link>
    </model>""" + sdf[end:]
    (worlds / f"{WORLD_3D}.sdf").write_text(sdf, encoding="utf-8")
    # The same surface on the onboard-map grid, for the localizer (DSM
    # projection): metres above the ground at the world origin.
    georef_path = scene_json.parent / "orthophoto_georef.json"
    if georef_path.exists():
        g = json.loads(georef_path.read_text())
        img = cv2.imread(str(scene_json.parent / "orthophoto.png"), cv2.IMREAD_GRAYSCALE)
        gsd = g["matrix_ne_m_per_px"][1][0]
        rows, cols = np.mgrid[0:img.shape[0], 0:img.shape[1]].astype(np.float64)
        surf = sample(dsm, e0 + g["origin_ne_m"][1] + gsd * cols, n0 + g["origin_ne_m"][0] - gsd * rows) - ground0
        np.save(scene_json.parent / "orthophoto_surface.npy", surf.astype(np.float32))
    info = {"world": WORLD_3D, "step_m": step_m, "ground_at_origin_m": ground0,
            "surface_min_m": float(z.min()), "surface_max_m": float(z.max()),
            "obj": str(obj), "dsm": dsm_path.name, "dtm": dtm_path.name}
    (scene_json.parent / "scene_3d.json").write_text(json.dumps(info, indent=2))
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("px4_dir", type=Path)
    ap.add_argument("--dsm", type=Path, required=True)
    ap.add_argument("--dtm", type=Path, required=True)
    ap.add_argument("--scene-json", type=Path, required=True, help="scene.json written by swiss_scene.py")
    ap.add_argument("--step-m", type=float, default=1.0)
    args = ap.parse_args()
    print(json.dumps(build(args.px4_dir.resolve(), args.dsm, args.dtm, args.scene_json, step_m=args.step_m),
                     indent=2))


if __name__ == "__main__":
    main()
