#!/usr/bin/env python3
"""Real-data check of the camera VO velocity against a PPK reference.

Inputs: the manifest of tools/munfrl_bag_slice.py and the output of
``python -m src.run_camera_odometry`` on it. For every accepted frame pair
the PPK velocity over EXACTLY the same pair interval is formed from the
reference positions of both frames.

Split in time: the CALIBRATION part (t < --split-s) estimates
  1. a constant heading offset between the attitude source and true north
     (circular median over intervals faster than 2 m/s; the MUN-FRL Xsens
     yaw is not north-referenced),
  2. a speed scale (flat-ground height model over uneven terrain),
  3. the noise model of the 'camera' profile (ofnav::flowVelocitySigma)
       sigma_v^2 = sigma_min^2 + (h*sigma_rate)^2 + (|v|*sigma_h_rel)^2
     by non-negative least squares on |e|^2/2 per interval;
and the VALIDATION part (t >= --split-s) only checks them: RMSE, mean NEES
(expected 2) and the fraction of NEES <= chi2(2, 95 %) for both the design
constants of the profile and the fitted ones. Adjacent intervals share a
frame (correlated errors): one slice is a check, not a sensor certificate.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

CHI2_2_95 = 5.991464547107979
PROFILE = {"sigma_min_m_s": 0.05, "sigma_rate_rad_s": 0.0015, "sigma_h_rel": 0.01}  # makeProfileConfig(camera)


def load(manifest: Path, vo: Path):
    man = list(csv.DictReader(manifest.open()))
    index = {r["timestamp_us"]: i for i, r in enumerate(man)}
    rows = []
    for r in csv.DictReader(vo.open()):
        if r["accepted"] != "1":
            continue
        i = index[r["timestamp_us"]]
        if i == 0:
            continue
        a, b = man[i - 1], man[i]
        dt = (int(b["timestamp_us"]) - int(a["timestamp_us"])) * 1e-6
        ref = np.array([(float(b["true_n_m"]) - float(a["true_n_m"])) / dt,
                        (float(b["true_e_m"]) - float(a["true_e_m"])) / dt])
        rows.append({
            "t": (int(b["timestamp_us"]) - int(man[0]["timestamp_us"])) * 1e-6,
            "vo": np.array([float(r["vn_m_s"]), float(r["ve_m_s"])]),
            "ref": ref,
            "h": 0.5 * (float(a["height_agl_m"]) + float(b["height_agl_m"])),
            "dt": dt,
        })
    return rows


def rot(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s], [s, c]])


def sigma2(p: dict, h: float, v: float) -> float:
    return p["sigma_min_m_s"] ** 2 + (h * p["sigma_rate_rad_s"]) ** 2 + (v * p["sigma_h_rel"]) ** 2


def metrics(rows, heading_deg: float, scale: float, params: dict | None) -> dict:
    if not rows:
        return {"intervals": 0}
    e = np.array([rot(-heading_deg) @ r["vo"] / scale - r["ref"] for r in rows])
    out = {
        "intervals": len(rows),
        "rmse_m_s": float(np.sqrt(np.mean(np.sum(e * e, axis=1)))),
        "median_abs_m_s": float(np.median(np.linalg.norm(e, axis=1))),
        "mean_error_ne_m_s": e.mean(axis=0).round(3).tolist(),
        "mean_height_m": float(np.mean([r["h"] for r in rows])),
        "mean_speed_m_s": float(np.mean([np.linalg.norm(r["ref"]) for r in rows])),
    }
    if params:
        nees = np.array([float(x @ x) / sigma2(params, r["h"], float(np.linalg.norm(r["ref"])))
                         for x, r in zip(e, rows)])
        out["mean_nees"] = float(nees.mean())
        out["nees_le_chi2_95_fraction"] = float(np.mean(nees <= CHI2_2_95))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("manifest", type=Path)
    ap.add_argument("vo_csv", type=Path)
    ap.add_argument("--split-s", type=float, default=12.0)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    rows = load(args.manifest, args.vo_csv)
    cal = [r for r in rows if r["t"] < args.split_s]
    val = [r for r in rows if r["t"] >= args.split_s]
    fast = [r for r in cal if np.linalg.norm(r["ref"]) > 2.0]
    if len(fast) < 5:
        raise SystemExit("too few fast calibration intervals for the heading offset")
    diffs = [math.atan2(r["vo"][1], r["vo"][0]) - math.atan2(r["ref"][1], r["ref"][0]) for r in fast]
    heading = math.degrees(math.atan2(np.median(np.sin(diffs)), np.median(np.cos(diffs))))
    scale = float(np.median([np.linalg.norm(r["vo"]) / np.linalg.norm(r["ref"]) for r in fast]))

    # Non-negative least squares on |e|^2 / 2 = s_min^2 + h^2 s_rate^2 + v^2 s_h^2.
    e2 = np.array([float(np.sum((rot(-heading) @ r["vo"] / scale - r["ref"]) ** 2)) / 2. for r in cal])
    a = np.array([[1., r["h"] ** 2, float(np.linalg.norm(r["ref"])) ** 2] for r in cal])
    active = [0, 1, 2]
    coef = np.zeros(3)
    while active:
        sol, *_ = np.linalg.lstsq(a[:, active], e2, rcond=None)
        if (sol >= 0).all():
            coef[active] = sol
            break
        active.pop(int(np.argmin(sol)))
    fitted = {"sigma_min_m_s": math.sqrt(coef[0]), "sigma_rate_rad_s": math.sqrt(coef[1]),
              "sigma_h_rel": math.sqrt(coef[2])}
    dts = [r["dt"] for r in rows]
    report = {
        "type": "MUNFRL_REAL_DATA_VO_VELOCITY_CHECK",
        "reference": "PPK (RTKLIB fix) positions of the same frame pair",
        "split_s": args.split_s,
        "pair_interval_s": float(np.median(dts)),
        "calibration": {
            "heading_offset_deg": heading,
            "speed_scale_vo_over_ref": scale,
            "fitted_noise_model": fitted,
            "raw_before_heading_and_scale": metrics(cal, 0.0, 1.0, None),
            "after": metrics(cal, heading, scale, fitted),
        },
        "validation": {
            "profile_constants": PROFILE,
            "with_profile_constants": metrics(val, heading, scale, PROFILE),
            "with_fitted_model": metrics(val, heading, scale, fitted),
            "without_scale_correction": metrics(val, heading, 1.0, fitted),
        },
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text)


if __name__ == "__main__":
    main()
