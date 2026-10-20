#!/usr/bin/env python3
"""Measure the camera frame time shift from SITL flight logs.

The VO velocity EKF2 received (``vehicle_visual_odometry``, body FRD,
stamped with ``timestamp_sample``) is rotated to N/E with the true yaw and
compared with the Gazebo ground-truth velocity at ``timestamp_sample +
shift`` for shifts in --range-ms. Only accelerating samples (true
horizontal acceleration > --min-accel) are used: at constant velocity the
shift is not observable. Errors are clipped at 1 m/s (VO outliers). The
shift is the minimum of the error curve averaged over the flights.

Series M/P calibration flights (3D swiss scene, 50 m): 105 ms (per-flight
minima: median 107 ms, IQR 88-146 ms). The VO velocity matches the true
velocity that much after its stamp: the frame stamps are early. Use on
calibration flights only; the result goes to the companion as
``--frame-time-shift-ms``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _report(d: Path) -> dict:
    for name in ("gz_map_companion_calibrate.json", "gz_map_companion_test.json", "gz_vo_companion_flight.json"):
        if (d / name).exists():
            return json.loads((d / name).read_text())
    raise FileNotFoundError(f"no flight report in {d}")


def error_curve(flight_dir: Path, lags_ms: np.ndarray, min_accel: float = 0.4) -> np.ndarray:
    from pyulog import ULog
    rep = _report(flight_dir)
    u = ULog(rep["ulog"], ["vehicle_visual_odometry", "vehicle_local_position_groundtruth",
                           "vehicle_attitude_groundtruth"])
    g = {x.name: x.data for x in u.data_list}
    vo, gt, a = g["vehicle_visual_odometry"], g["vehicle_local_position_groundtruth"], g["vehicle_attitude_groundtruth"]
    ok = np.isfinite(vo["velocity[0]"])
    t = vo["timestamp_sample"][ok].astype(float)
    q = np.column_stack([np.interp(t, a["timestamp"], a[f"q[{i}]"]) for i in range(4)])
    w, x, y, z = q.T
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    c, s = np.cos(yaw), np.sin(yaw)
    vx, vy = vo["velocity[0]"][ok], vo["velocity[1]"][ok]
    vn, ve = c * vx - s * vy, s * vx + c * vy
    tg = gt["timestamp"].astype(float)
    acc = np.hypot(np.gradient(gt["vx"], tg / 1e6), np.gradient(gt["vy"], tg / 1e6))
    m = np.interp(t, tg, acc) > min_accel
    return np.array([np.mean(np.minimum(np.hypot(vn[m] - np.interp(t[m] + lag * 1e3, tg, gt["vx"]),
                                                 ve[m] - np.interp(t[m] + lag * 1e3, tg, gt["vy"])), 1.0))
                     for lag in lags_ms])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("flights", type=Path, nargs="+")
    ap.add_argument("--range-ms", type=float, nargs=2, default=(-100.0, 300.0))
    ap.add_argument("--step-ms", type=float, default=5.0)
    ap.add_argument("--min-accel", type=float, default=0.4)
    args = ap.parse_args(argv)
    lags = np.arange(args.range_ms[0], args.range_ms[1] + 1e-9, args.step_ms)
    curves = [error_curve(f, lags, args.min_accel) for f in args.flights]
    mean = np.mean(curves, axis=0)
    per = [float(lags[np.argmin(c)]) for c in curves]
    out = {"frame_time_shift_ms": float(lags[np.argmin(mean)]),
           "mean_clipped_error_at_shift_m_s": float(mean.min()),
           "mean_clipped_error_at_zero_m_s": float(np.interp(0.0, lags, mean)),
           "per_flight_minimum_ms": dict(zip((str(f) for f in args.flights), per)),
           "per_flight_median_ms": float(np.median(per))}
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
