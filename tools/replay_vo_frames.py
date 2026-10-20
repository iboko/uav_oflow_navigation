#!/usr/bin/env python3
"""Replay the companion's camera VO offline on recorded frames (SITL ablations).

Input: a flight directory of ``gz_map_companion_flight.py`` flown with
``--save-every 1`` (every camera frame in ``frames/``, named by companion
time; the report's ``time_offset_us`` moves them to the autopilot clock).
The VO is the companion's ``ContinuousPlanarOdometry`` with the same
settings; its inputs are chosen per run:

  --attitude ekf|truth      roll/pitch/yaw for rotation compensation
  --height range|truth      scale height (range: rangefinder, as onboard)
  --depth none|dsm          one height per frame, or each feature's depth on the DSM
  --position ekf|truth      horizontal position for the DSM lookup
  --altitude range|ekf|truth  camera altitude above the DSM datum
                            (range: rangefinder + DSM at the nadir)
  --pair-step N             pair every N-th frame (longer baseline)

The reference is the ground-truth displacement over each pair interval
divided by its length (the mean velocity the VO measures). Only the flight
segment of the report is scored. Diagnostic only; simulated flights.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.camera_rotation_compensation import CameraMountCalibration  # noqa: E402
from src.ground_visual_motion import CameraCalibration  # noqa: E402
from src.continuous_visual_odometry import ContinuousPlanarOdometry, VisualFrame  # noqa: E402
from src.surface_depth import SurfaceDepth  # noqa: E402
from src.velocity_uncertainty import VelocityErrorAssumptions  # noqa: E402

# As in tools/gz_vo_companion.py (that module needs the gz Python bindings).
IMAGE_TO_BODY = ((0.0, -1.0), (1.0, 0.0))
BODY_FROM_CAMERA = ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))


def calibration(width: int, height: int, hfov_rad: float) -> CameraCalibration:
    f = 0.5 * width / math.tan(0.5 * hfov_rad)
    return CameraCalibration(f, f, 0.5 * width, 0.5 * height, IMAGE_TO_BODY)


def _report(d: Path) -> dict:
    for name in ("gz_map_companion_calibrate.json", "gz_map_companion_test.json"):
        if (d / name).exists():
            return json.loads((d / name).read_text())
    raise FileNotFoundError(f"no flight report in {d}")


def quat_rpy(q: np.ndarray):
    w, x, y, z = q.T
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def interp_quat(t, data) -> np.ndarray:
    q = np.column_stack([np.interp(t, data["timestamp"], data[f"q[{i}]"]) for i in range(4)])
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def load_inputs(flight: Path, surface: SurfaceDepth | None):
    from pyulog import ULog
    rep = _report(flight)
    offset = int(rep["companion_final_status"].get("time_offset_us") or 0)
    frames = sorted((flight / "frames").glob("*.png"), key=lambda p: int(p.stem))
    t = np.array([int(p.stem) + offset for p in frames], dtype=float)
    u = ULog(rep["ulog"], ["vehicle_attitude", "vehicle_attitude_groundtruth", "distance_sensor",
                           "vehicle_local_position", "vehicle_local_position_groundtruth"])
    g = {x.name: x.data for x in u.data_list}
    ds, lp, gt = g["distance_sensor"], g["vehicle_local_position"], g["vehicle_local_position_groundtruth"]
    ip = lambda d, k: np.interp(t, d["timestamp"], d[k])  # noqa: E731
    inp = {"t": t, "frames": frames, "rep": rep,
           "rpy_ekf": np.column_stack(quat_rpy(interp_quat(t, g["vehicle_attitude"]))),
           "rpy_truth": np.column_stack(quat_rpy(interp_quat(t, g["vehicle_attitude_groundtruth"]))),
           "range": ip(ds, "current_distance"),
           "ne_ekf": np.column_stack([ip(lp, "x"), ip(lp, "y")]), "z_ekf": ip(lp, "z"),
           "ne_truth": np.column_stack([ip(gt, "x"), ip(gt, "y")]), "z_truth": ip(gt, "z"),
           "gt": gt}
    r, p = inp["rpy_ekf"][:, 0], inp["rpy_ekf"][:, 1]
    inp["range_v"] = inp["range"] * np.cos(r) * np.cos(p)
    if surface is not None:
        # DSM datum of the local frame: rangefinder + surface at the nadir
        # minus the local altitude, median over the flight (truth position).
        nadir = surface.height_at(inp["ne_truth"][:, 0], inp["ne_truth"][:, 1])
        c = float(np.median(inp["range_v"] + nadir + inp["z_truth"]))
        inp["datum_m"] = c
        inp["alt_truth"] = -inp["z_truth"] + c
        inp["alt_ekf"] = -inp["z_ekf"] + c
        inp["alt_range"] = inp["range_v"] + surface.height_at(inp["ne_ekf"][:, 0], inp["ne_ekf"][:, 1])
        inp["h_truth"] = inp["alt_truth"] - nadir
    else:
        inp["h_truth"] = -inp["z_truth"]
    return inp


def replay(inp: dict, *, attitude="ekf", height="range", depth="none", position="ekf", altitude="range",
           pair_step=1, surface: SurfaceDepth | None = None, hfov=1.2) -> dict:
    first = cv2.imread(str(inp["frames"][0]), cv2.IMREAD_GRAYSCALE)
    h, w = first.shape
    cal = calibration(w, h, hfov)
    vo = ContinuousPlanarOdometry(
        cal, camera_mount=CameraMountCalibration(BODY_FROM_CAMERA), camera_offset_body_m=(0.0, 0.0, 0.1),
        velocity_error_assumptions=VelocityErrorAssumptions(
            fitted_displacement_sigma_px=0.5, height_sigma_m=0.05, interval_sigma_s=0.002,
            yaw_sigma_rad=0.02, focal_relative_sigma=0.01, velocity_model_floor_m_s=0.05),
        max_tilt_rad=0.25, max_pair_dt_s=min(1.0, 0.25 * pair_step), surface=surface if depth == "dsm" else None)
    rpy = inp[f"rpy_{attitude}"]
    hh = inp["range_v"] if height == "range" else inp["h_truth"]
    ne = inp[f"ne_{position}"]
    alt = inp.get(f"alt_{altitude}")
    gt = inp["gt"]
    t0, t1 = (inp["rep"].get("segments") or {}).get("flight") or (inp["t"][0], inp["t"][-1])
    errs, status = [], {}
    for k in range(0, len(inp["t"]), pair_step):
        tk = int(inp["t"][k])
        if hh[k] < 0.5:
            vo.reset()
            continue
        gray = cv2.imread(str(inp["frames"][k]), cv2.IMREAD_GRAYSCALE)
        frame = VisualFrame(tk, gray, float(hh[k]), float(rpy[k, 2]), float(rpy[k, 0]), float(rpy[k, 1]),
                            (float(ne[k, 0]), float(ne[k, 1])) if depth == "dsm" else None,
                            float(alt[k]) if depth == "dsm" else None)
        r = vo.process(frame)
        status[r.status] = status.get(r.status, 0) + 1
        if not r.accepted or not t0 <= tk <= t1:
            continue
        a, b = r.interval_start_us, r.timestamp_us
        dt = (b - a) * 1e-6
        ref = [(np.interp(b, gt["timestamp"], gt[c]) - np.interp(a, gt["timestamp"], gt[c])) / dt for c in "xy"]
        errs.append((b, r.vn_m_s - ref[0], r.ve_m_s - ref[1], ref[0], ref[1]))
    e = np.array(errs)
    mag = np.hypot(e[:, 1], e[:, 2])
    core = mag < 1.0
    return {"pairs": len(e), "status": status,
            "rms_m_s": float(np.sqrt(np.mean(mag ** 2))),
            "core_axis_std_m_s": float(np.std(np.r_[e[core, 1], e[core, 2]])),
            "core_axis_mean_m_s": [float(np.mean(e[core, 1])), float(np.mean(e[core, 2]))],
            "outliers_over_1m_s": int((~core).sum()), "p95_m_s": float(np.percentile(mag, 95)),
            "errors": e}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("flight", type=Path)
    ap.add_argument("--attitude", choices=("ekf", "truth"), default="ekf")
    ap.add_argument("--height", choices=("range", "truth"), default="range")
    ap.add_argument("--depth", choices=("none", "dsm"), default="none")
    ap.add_argument("--position", choices=("ekf", "truth"), default="ekf")
    ap.add_argument("--altitude", choices=("range", "ekf", "truth"), default="range")
    ap.add_argument("--pair-step", type=int, default=1)
    ap.add_argument("--surface", type=Path, help="DSM .npy on the orthophoto grid (needed for --depth dsm)")
    ap.add_argument("--georef", type=Path, help="orthophoto georef JSON of the surface grid")
    ap.add_argument("--out", type=Path, help="write per-pair errors (CSV)")
    args = ap.parse_args(argv)
    surface = None
    if args.surface:
        surface = SurfaceDepth.from_georef(np.load(args.surface), json.loads(args.georef.read_text()))
    if args.depth == "dsm" and surface is None:
        ap.error("--depth dsm needs --surface and --georef")
    inp = load_inputs(args.flight, surface)
    res = replay(inp, attitude=args.attitude, height=args.height, depth=args.depth, position=args.position,
                 altitude=args.altitude, pair_step=args.pair_step, surface=surface)
    if args.out:
        np.savetxt(args.out, res["errors"], delimiter=",", header="t_us,err_n,err_e,ref_vn,ref_ve", comments="")
    res.pop("errors")
    res["datum_m"] = inp.get("datum_m")
    print(json.dumps(res, indent=1))
    return 0 if math.isfinite(res["rms_m_s"]) else 1


if __name__ == "__main__":
    sys.exit(main())
