#!/usr/bin/env python3
"""Calibrate the companion MSCKF velocity covariance on separate SITL flights.

Each flight directory comes from ``gz_vo_companion_flight.py --estimator
msckf`` (``companion_vo.csv`` with body-FRD velocity and variances on the
PX4 clock, ``gz_vo_companion_flight.json`` with the ulog path). Reference:
Gazebo ground-truth velocity rotated into the true body axes.

The first 3 flights fit, the rest are held out (``calibrate_velocity_
covariance``: maximum-likelihood scale of the horizontal covariance, equal
flight weights, NEES and 95 % coverage on the held-out flights). The
vertical axis gets its own scale (mean e^2/var on the fit flights) and the
same held-out check. Nothing is fitted on the held-out flights.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.velocity_covariance_calibration import (  # noqa: E402
    CHI2_2_95, VelocityErrorFlight, calibrate_velocity_covariance,
)

CHI2_1_95 = 3.841458820694124


def body_errors(flight_dir: Path, skip_s: float = 2.0):
    from pyulog import ULog
    report = json.loads((flight_dir / "gz_vo_companion_flight.json").read_text())
    u = ULog(report["ulog"], ["vehicle_local_position_groundtruth", "vehicle_attitude_groundtruth"])
    d = {x.name: x.data for x in u.data_list}
    g, a = d["vehicle_local_position_groundtruth"], d["vehicle_attitude_groundtruth"]
    rows = np.genfromtxt(flight_dir / "companion_vo.csv", delimiter=",", names=True)
    t = rows["t_us"]
    # Skip the first seconds after each (re)initialisation: the filter is
    # started from the alignment and its covariance is the alignment's.
    gap = np.diff(t, prepend=-1e12) > 0.5e6
    start = np.maximum.accumulate(np.where(gap, t, -np.inf))
    keep = t - start >= skip_s * 1e6
    t = t[keep]
    v_frd = np.column_stack([rows[k][keep] for k in ("vx", "vy", "vz")])
    var = np.column_stack([rows[k][keep] for k in ("var_x", "var_y", "var_z")])
    v_ned = np.column_stack([np.interp(t, g["timestamp"], g[k]) for k in ("vx", "vy", "vz")])
    q = np.column_stack([np.interp(t, a["timestamp"], a[f"q[{i}]"]) for i in range(4)])
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    r = np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
                  np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
                  np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], -2)
    truth_b = np.einsum("nji,nj->ni", r, v_ned)        # R^T v
    return v_frd - truth_b, var, int(np.sum(~keep))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("flights", type=Path, nargs="+", help="3 fit flights, then held-out flights")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if len(args.flights) < 5:
        ap.error("need 3 fit + at least 2 held-out flights")
    data = [body_errors(f) for f in args.flights]
    flights = [VelocityErrorFlight(f.name, tuple(map(tuple, e[:, :2])),
                                   tuple(((float(v[0]), 0.0), (0.0, float(v[1]))) for v in var))
               for f, (e, var, _) in zip(args.flights, data)]
    cal = calibrate_velocity_covariance(flights[:3], flights[3:])
    # Vertical: same procedure, one axis.
    ez = [(e[:, 2], var[:, 2]) for e, var, _ in data]
    bz = float(np.mean([np.mean(e) for e, _ in ez[:3]]))
    sz = float(np.mean([np.mean((e - bz) ** 2 / v) for e, v in ez[:3]]))
    hz = [((e - bz) ** 2 / (sz * v)) for e, v in ez[3:]]
    out = {
        "model": "variance = scale * MSCKF variance (horizontal body axes; vertical own scale)",
        "horizontal": cal.to_json(),
        "vertical": {"scale": sz, "bias_m_s": bz,
                     "holdout_mean_nees_1d": [round(float(np.mean(n)), 3) for n in hz],
                     "holdout_coverage_95pct": [round(float(np.mean(n <= CHI2_1_95)), 3) for n in hz]},
        "uncalibrated_mean_nees_2d": [round(float(np.mean(np.sum(e[:, :2] ** 2 / var[:, :2], axis=1))), 4)
                                      for e, var, _ in data],
        "samples": [len(e) for e, _, _ in data],
        "skipped_after_init": [s for _, _, s in data],
        "flights": [str(f) for f in args.flights],
        "note": "simulated flights (Gazebo); errors are time-correlated, flights are the sampling unit",
    }
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("uncalibrated_mean_nees_2d", "samples")}))
    print(json.dumps({"scale_xy": cal.scale, "holdout_mean_nees_2d": cal.holdout_mean_nees_2d,
                      "holdout_coverage": cal.holdout_coverage_95pct, "consistent": cal.consistent_on_holdout,
                      "vertical": out["vertical"]}))


if __name__ == "__main__":
    main()
