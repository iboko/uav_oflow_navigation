#!/usr/bin/env python3
"""Calibrate the VO-gap bridge (src/companion_map.py VelocityBridge).

Across a VO gap the map anchor can be carried with the autopilot's own
horizontal velocity (EKF2 on the IMU, without VO). The error of that
velocity integrated over the gap is modelled per axis as
sigma(L)^2 = s0^2 + (s1 L)^2 + (s2 L^2)^2 for a gap of L s. Here the three
parameters are fitted on calibration flights: least squares on the squared
axis errors (non-negative), equal weight per flight. Coverage and NEES are
reported leave-one-flight-out (fit on the others, evaluated on the held-out
one): nothing is fitted on the flight it is evaluated on.

Input: flight directories of ``gz_map_companion_flight.py`` (test mode,
GNSS off after check 2), e.g. with injected VO dropouts
(``--vo-outages-s``). Gaps: consecutive rows of ``companion_vo.csv`` more
than --min-gap-s apart, after GNSS loss, up to the end of the route
(or of the map fault phase, check 4, if later). The anchor's
gap runs from the end of one VO interval to the start of the next; the rows
are interval midpoints, so half the median row spacing is taken off each
side. Reference: Gazebo ground-truth position; autopilot velocity:
``vehicle_local_position`` vx, vy (same clock: SITL lockstep).

Simulated flights: not a sensor certification.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _report(d: Path) -> dict:
    for name in ("gz_map_companion_test.json", "gz_map_companion_heading.json"):
        if (d / name).exists():
            return json.loads((d / name).read_text())
    raise FileNotFoundError(f"no flight report in {d}")


def flight_gaps(d: Path, min_gap_s: float = 0.3) -> np.ndarray:
    """Rows (gap_s, err_n, err_e) of one flight."""
    from pyulog import ULog
    r = _report(d)
    t_lo = max(r["gnss_off_us"], r["route_start_us"])
    t_hi = max(r["route_end_us"], r.get("fault_end_us") or 0)
    rows = np.genfromtxt(d / "companion_vo.csv", delimiter=",", names=True)
    t = rows["t_us"][np.isfinite(rows["vn"])]
    half = 0.5 * float(np.median(np.diff(t)))
    u = ULog(r["ulog"], ["vehicle_local_position", "vehicle_local_position_groundtruth"])
    g = {x.name: x.data for x in u.data_list}
    lp, gt = g["vehicle_local_position"], g["vehicle_local_position_groundtruth"]
    out = []
    for a, b in zip(t[:-1], t[1:]):
        t0, t1 = a + half, b - half
        if (t1 - t0) * 1e-6 < min_gap_s or t0 < t_lo or t1 > t_hi:
            continue
        m = (lp["timestamp"] > t0) & (lp["timestamp"] < t1)
        knots = np.concatenate([[t0], lp["timestamp"][m], [t1]])
        if np.max(np.diff(knots)) > 200_000:
            continue
        d_est = [np.trapezoid(np.interp(knots, lp["timestamp"], lp[c]), knots * 1e-6) for c in ("vx", "vy")]
        d_true = [np.interp(t1, gt["timestamp"], gt[c]) - np.interp(t0, gt["timestamp"], gt[c]) for c in "xy"]
        out.append(((t1 - t0) * 1e-6, d_est[0] - d_true[0], d_est[1] - d_true[1]))
    return np.array(out).reshape(-1, 3)


def _nnls(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Non-negative least squares for a few columns: best over active sets."""
    from itertools import combinations
    best, best_res = np.zeros(a.shape[1]), float(b @ b)
    for k in range(1, a.shape[1] + 1):
        for cols in combinations(range(a.shape[1]), k):
            x, *_ = np.linalg.lstsq(a[:, cols], b, rcond=None)
            if np.all(x >= 0):
                full = np.zeros(a.shape[1])
                full[list(cols)] = x
                res = float(np.sum((a @ full - b) ** 2))
                if res < best_res:
                    best, best_res = full, res
    return best


def fit(flights: list[np.ndarray]) -> tuple[float, float, float]:
    """Non-negative least squares of e^2 on [1, L^2, L^4], equal flight weights."""
    a_rows, b_rows = [], []
    for f in flights:
        if len(f) == 0:
            continue
        w = 1.0 / np.sqrt(2 * len(f))          # each flight weighs the same (two axes per gap)
        for col in (1, 2):
            a_rows.append(w * np.column_stack([np.ones(len(f)), f[:, 0] ** 2, f[:, 0] ** 4]))
            b_rows.append(w * f[:, col] ** 2)
    coef = _nnls(np.vstack(a_rows), np.concatenate(b_rows))
    return tuple(float(np.sqrt(c)) for c in coef)


def sigma(params, gap_s):
    s0, s1, s2 = params
    return np.sqrt(s0 ** 2 + (s1 * gap_s) ** 2 + (s2 * gap_s ** 2) ** 2)


def evaluate(params, f: np.ndarray) -> dict:
    s = sigma(params, f[:, 0])
    nees = (f[:, 1] ** 2 + f[:, 2] ** 2) / s ** 2
    return {"gaps": int(len(f)), "mean_nees_2d": round(float(np.mean(nees)), 3),
            "coverage_95": round(float(np.mean(nees <= 5.991)), 3),
            "max_gap_s": round(float(f[:, 0].max()), 2)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("flights", type=Path, nargs="+")
    ap.add_argument("--min-gap-s", type=float, default=0.3)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if len(args.flights) < 2:
        ap.error("need at least two calibration flights (leave-one-out evaluation)")
    data = [flight_gaps(d, args.min_gap_s) for d in args.flights]
    if any(len(f) == 0 for f in data):
        ap.error("a flight has no VO gaps after GNSS loss: " +
                 ", ".join(str(d) for d, f in zip(args.flights, data) if len(f) == 0))
    params = fit(data)
    held_out = {}
    for i, d in enumerate(args.flights):
        p = fit(data[:i] + data[i + 1:])
        held_out[d.name] = evaluate(p, data[i]) | {"fit_without": [round(x, 4) for x in p]}
    allg = np.vstack(data)
    bins = []
    for lo, hi in ((0.3, 1), (1, 2), (2, 4), (4, 8), (8, 30)):
        m = (allg[:, 0] >= lo) & (allg[:, 0] < hi)
        if m.any():
            bins.append({"gap_s": [lo, hi], "gaps": int(m.sum()),
                         "rms_axis_m": round(float(np.sqrt(np.mean(allg[m, 1:] ** 2))), 3),
                         "model_sigma_m": round(float(np.mean(sigma(params, allg[m, 0]))), 3)})
    out = {"sigma0_m": params[0], "sigma_per_s_m": params[1], "sigma_per_s2_m": params[2],
           "flights": [str(d) for d in args.flights], "all": evaluate(params, allg),
           "held_out": held_out, "bins": bins,
           "note": "VO-gap bridge with the autopilot velocity; simulated flights, not a sensor certification"}
    args.out.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
