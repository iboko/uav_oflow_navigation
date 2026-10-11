#!/usr/bin/env python3
"""Tightly coupled VIO (MSCKF) on MUN-FRL slices against PPK.

Same slices, IMU time model, camera-IMU offset (-37 ms), initial state and
evaluation as tools/munfrl_vo_eskf.py (loosely coupled VO + 15-state ESKF):
the heading of the Xsens frame is aligned on the FIRST half of a slice,
drift and velocity errors are measured on the SECOND half. Height above
ground, rangefinder and PPK are NOT used by the filter.

Usage: munfrl_msckf.py NAME:SLICE_DIR [...] --out report.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from munfrl_vo_eskf import IMU_NOISE, evaluate, load_attitude, load_imu, procrustes  # noqa: E402
from src.feature_tracker import KltTracker  # noqa: E402
from src.imu_preintegration import ImuReading  # noqa: E402
from src.msckf_vio import Msckf, PinholeCamera  # noqa: E402

CAMERA_TO_IMU_OFFSET_US = -37_000  # t_imu = t_camera + offset (docs/REAL_DATA_MUNFRL_RU.md)


def load_manifest(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as fp:
        rows = list(csv.DictReader(fp))
    return [{"t": int(r["timestamp_us"]), "image": r["image_path"], "h": float(r["height_agl_m"]),
             "ref_pos": (float(r["true_n_m"]), float(r["true_e_m"])),
             "ref_vel": (float(r["true_vn_m_s"]), float(r["true_ve_m_s"])),
             "ppk_fix": str(r.get("ppk_quality", "1")).strip() == "1"} for r in rows]


def interpolate(a: ImuReading, b: ImuReading, t_us: int) -> ImuReading:
    w = (t_us - a.timestamp_us) / (b.timestamp_us - a.timestamp_us)
    mix = lambda x, y: tuple((1 - w) * np.asarray(x) + w * np.asarray(y))  # noqa: E731
    return ImuReading(t_us, mix(a.gyro_body_rad_s, b.gyro_body_rad_s),
                      mix(a.specific_force_body_m_s2, b.specific_force_body_m_s2))


def run(slice_dir: Path, *, pixel_sigma: float, max_clones: int, max_features: int,
        imu_decimate: int, offset_us: int, init_velocity: str = "gnss") -> dict:
    cam_cfg = json.loads((slice_dir / "camera.yaml").read_text())
    camera = PinholeCamera(cam_cfg["focal_x_px"], cam_cfg["focal_y_px"], cam_cfg["center_x_px"],
                           cam_cfg["center_y_px"], tuple(map(tuple, cam_cfg["body_from_camera"])),
                           tuple(cam_cfg["camera_offset_body_m"]), pixel_sigma=pixel_sigma)
    imu, imu_info = load_imu(slice_dir / "imu.csv", decimate=imu_decimate)
    frames = [f for f in load_manifest(slice_dir / "manifest.csv")
              if imu[0].timestamp_us < f["t"] + offset_us < imu[-1].timestamp_us]
    q0 = load_attitude(slice_dir / "attitude.csv", imu[0].timestamp_us)
    # The VIO starts when GNSS is lost: velocity from GNSS (here PPK at the
    # first frame, sigma 0.3 m/s), position origin arbitrary. With a zero
    # velocity guess (sigma 5 m/s, as the loosely coupled run) the first
    # low-parallax tracks set the velocity through a poor depth
    # linearisation and the filter diverges (checked on quarry2).
    if init_velocity == "gnss":
        v0, sv = (*frames[0]["ref_vel"], 0.0), 0.3
    else:
        v0, sv = (0.0, 0.0, 0.0), 5.0
    sig = (0.01, 0.01, 0.01, sv, sv, sv, 0.05, 0.05, 0.2, 0.01, 0.01, 0.01, 0.3, 0.3, 0.3)
    filt = Msckf(camera=camera, noise=IMU_NOISE, quaternion_body_to_ned=q0, initial_error_sigma=sig,
                 initial_velocity_ned_m_s=v0, max_clones=max_clones)
    tracker = KltTracker(max_features=max_features)
    k, traj, start = 0, [], time.perf_counter()
    filt.predict(imu[0])
    k = 1
    for f in frames:
        t_imu = f["t"] + offset_us
        while k < len(imu) and imu[k].timestamp_us < t_imu:
            filt.predict(imu[k])
            k += 1
        if filt.last_imu.timestamp_us != t_imu:
            filt.predict(interpolate(filt.last_imu, imu[k], t_imu))
        gray = cv2.imread(str(slice_dir / f["image"]), cv2.IMREAD_GRAYSCALE)
        feats = tracker.track(gray)
        filt.process_frame(t_imu, feats)
        traj.append({"t": f["t"], "p": tuple(filt.p[:2]), "v": tuple(filt.v[:2]),
                     "pv": (filt.cov[3, 3], filt.cov[4, 4])})
    elapsed = time.perf_counter() - start
    half = len(frames) // 2
    theta, _ = procrustes(np.array([x["v"] for x in traj[:half]]), np.array([f["ref_vel"] for f in frames[:half]]))
    metrics = evaluate(frames, traj, theta)
    stats = dict(vars(filt.stats))
    nis = np.asarray(stats.pop("nis_per_dof"))
    stats["nis_per_dof_median"] = float(np.median(nis)) if len(nis) else None
    stats["nis_per_dof_mean"] = float(np.mean(nis)) if len(nis) else None
    # Pixel sigma that would make the mean NIS/dof equal to 1 (for calibration).
    stats["pixel_sigma_for_unit_nis"] = float(pixel_sigma * np.sqrt(np.mean(nis))) if len(nis) else None
    return {"imu": imu_info, "frames": len(frames), "heading_alignment_deg": math.degrees(theta),
            "init_velocity": init_velocity, "stats": stats, "tracker": tracker.counts,
            "final_gyro_bias": filt.bg.round(5).tolist(), "final_accel_bias": filt.ba.round(4).tolist(),
            "runtime_s_per_frame": elapsed / len(frames), **metrics}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("slices", nargs="+", help="NAME:SLICE_DIR")
    ap.add_argument("--pixel-sigma", type=float, default=1.0)
    ap.add_argument("--max-clones", type=int, default=10)
    ap.add_argument("--max-features", type=int, default=150)
    ap.add_argument("--imu-decimate", type=int, default=4, help="400 Hz Xsens -> 100 Hz")
    ap.add_argument("--camera-offset-us", type=int, default=CAMERA_TO_IMU_OFFSET_US)
    ap.add_argument("--init-velocity", choices=("gnss", "zero"), default="gnss")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    report = {"type": "MUNFRL_MSCKF_VS_PPK", "pixel_sigma": args.pixel_sigma, "max_clones": args.max_clones,
              "camera_to_imu_offset_us": args.camera_offset_us, "flights": {}}
    for spec in args.slices:
        name, d = spec.split(":")
        report["flights"][name] = run(Path(d), pixel_sigma=args.pixel_sigma, max_clones=args.max_clones,
                                      max_features=args.max_features, imu_decimate=args.imu_decimate,
                                      offset_us=args.camera_offset_us, init_velocity=args.init_velocity)
        r = report["flights"][name]
        print(name, {k: (round(r[k], 3) if isinstance(r[k], float) else r[k]) for k in
                     ("final_drift_m", "max_drift_m", "path_length_m", "velocity_rmse_m_s", "velocity_nees_2d",
                      "runtime_s_per_frame")}, "sigma_for_unit_nis", r["stats"]["pixel_sigma_for_unit_nis"], flush=True)
    if args.out:
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
