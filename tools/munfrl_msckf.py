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
from src.imu_preintegration import ImuBias  # noqa: E402
from src.msckf_vio import Msckf, PinholeCamera  # noqa: E402
from src.vi_initialization import initialize  # noqa: E402

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
    tracker = KltTracker(max_features=max_features)
    start = time.perf_counter()
    filt, init_report = None, None
    if init_velocity in ("gnss", "zero"):
        if init_velocity == "gnss":
            v0, sv = (*frames[0]["ref_vel"], 0.0), 0.3
        else:
            v0, sv = (0.0, 0.0, 0.0), 5.0
        sig = (0.01, 0.01, 0.01, sv, sv, sv, 0.05, 0.05, 0.2, 0.01, 0.01, 0.01, 0.3, 0.3, 0.3)
        filt = Msckf(camera=camera, noise=IMU_NOISE, quaternion_body_to_ned=q0, initial_error_sigma=sig,
                     initial_velocity_ned_m_s=v0, max_clones=max_clones)
        filt.predict(imu[0])
    window, window_heights, last_try = [], [], 0
    traj = []
    k = 1
    for f in frames:
        t_imu = f["t"] + offset_us
        gray = cv2.imread(str(slice_dir / f["image"]), cv2.IMREAD_GRAYSCALE)
        feats = tracker.track(gray)
        if filt is None:
            # Without GNSS: linear VI initialisation over a growing window,
            # 1.5 s at least, tried every 0.5 s up to 5 s.
            window.append((t_imu, feats))
            # vi_height: PPK height above the take-off point stands in for a
            # rangefinder (MUN-FRL has none); flat ground assumed.
            window_heights.append((t_imu, f["h"]))
            span = (t_imu - window[0][0]) * 1e-6
            if span >= 1.5 and t_imu - last_try >= 500_000:
                last_try = t_imu
                q_ahrs = load_attitude(slice_dir / "attitude.csv", window[0][0])
                heights = {t: h for t, h in window_heights} if init_velocity == "vi_height" else None
                res = initialize(imu, window, camera, q_ahrs, estimate_accel_bias=True, heights_agl_m=heights)
                init_report = {"status": res.status, "window_s": round(span, 2), "features": res.features,
                               "sigma_velocity_m_s": res.sigma_velocity_m_s,
                               "sigma_accel_bias_m_s2": res.sigma_accel_bias_m_s2,
                               "velocity_end": list(res.velocity_end_ned_m_s),
                               "ppk_velocity_at_end": list(f["ref_vel"]),
                               "accel_bias": list(res.accel_bias_body_m_s2),
                               "median_depth_m": res.median_depth_m, "chi2_per_dof": res.chi2_per_dof,
                               "gravity_tilt_deg": list(res.gravity_tilt_deg),
                               "excitation_m_s2": res.excitation_m_s2,
                               "ppk_height_m": f["h"]}
                if res.ok:
                    sv = np.sqrt(np.clip(np.diag(np.asarray(res.velocity_end_cov)), 1e-4, None))
                    sig = (0.01, 0.01, 0.01, *sv, 0.05, 0.05, 0.2, 0.01, 0.01, 0.01,
                           *(max(res.sigma_accel_bias_m_s2, 0.02),) * 3)
                    filt = Msckf(camera=camera, noise=IMU_NOISE, quaternion_body_to_ned=res.quaternion_end,
                                 initial_error_sigma=sig, initial_velocity_ned_m_s=res.velocity_end_ned_m_s,
                                 initial_bias=ImuBias(accel_body_m_s2=res.accel_bias_body_m_s2),
                                 max_clones=max_clones)
                    while k < len(imu) and imu[k].timestamp_us < t_imu:
                        k += 1
                    filt.predict(interpolate(imu[k - 1], imu[k], t_imu))
                elif span > 5.0:
                    return {"frames": len(frames), "init_velocity": init_velocity, "vi_init": init_report,
                            "failed": "no consistent VI initialisation within 5 s"}
            traj.append({"t": f["t"], "p": (math.nan, math.nan), "v": (math.nan, math.nan),
                         "pv": (math.nan, math.nan)})
            continue
        while k < len(imu) and imu[k].timestamp_us < t_imu:
            filt.predict(imu[k])
            k += 1
        if filt.last_imu.timestamp_us != t_imu:
            filt.predict(interpolate(filt.last_imu, imu[k], t_imu))
        filt.process_frame(t_imu, feats)
        traj.append({"t": f["t"], "p": tuple(filt.p[:2]), "v": tuple(filt.v[:2]),
                     "pv": (filt.cov[3, 3], filt.cov[4, 4])})
    elapsed = time.perf_counter() - start
    half = len(frames) // 2
    first = [i for i in range(half) if np.isfinite(traj[i]["v"]).all()]
    theta, _ = procrustes(np.array([traj[i]["v"] for i in first]), np.array([frames[i]["ref_vel"] for i in first]))
    metrics = evaluate(frames, traj, theta)
    stats = dict(vars(filt.stats))
    nis = np.asarray(stats.pop("nis_per_dof"))
    stats["nis_per_dof_median"] = float(np.median(nis)) if len(nis) else None
    stats["nis_per_dof_mean"] = float(np.mean(nis)) if len(nis) else None
    # Pixel sigma that would make the mean NIS/dof equal to 1 (for calibration).
    stats["pixel_sigma_for_unit_nis"] = float(pixel_sigma * np.sqrt(np.mean(nis))) if len(nis) else None
    return {"imu": imu_info, "frames": len(frames), "heading_alignment_deg": math.degrees(theta),
            "init_velocity": init_velocity, "vi_init": init_report, "stats": stats, "tracker": tracker.counts,
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
    ap.add_argument("--init-velocity", choices=("gnss", "zero", "vi", "vi_height"), default="gnss",
                    help="vi: visual-inertial initialisation, no GNSS; vi_height: plus height above "
                         "ground (PPK height as a rangefinder stand-in)")
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
        if "failed" in r:
            print(name, "FAILED", r["vi_init"], flush=True)
            continue
        print(name, {k: (round(r[k], 3) if isinstance(r[k], float) else r[k]) for k in
                     ("final_drift_m", "max_drift_m", "path_length_m", "velocity_rmse_m_s", "velocity_nees_2d",
                      "runtime_s_per_frame")}, "sigma_for_unit_nis", r["stats"]["pixel_sigma_for_unit_nis"], flush=True)
    if args.out:
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
