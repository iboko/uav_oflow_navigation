#!/usr/bin/env python3
"""Calibrate the camera-VO velocity covariance on separate SITL flights.

The VO error model (``velocity_uncertainty``) is built from assumed error
sources. At 50 m it gives sigma ~1.6 m/s per horizontal axis, while the
velocity error on the 3D swiss scene is ~0.12 m/s: EKF2 then barely uses
VO, takes the velocity from 2-Hz map positions with a lag, and the hold
can oscillate (series P, VO-gate test diag). Here the horizontal model
variance is scaled by a maximum-likelihood factor fitted on calibration
flights, with the same procedure as the MSCKF calibration
(``calibrate_velocity_covariance``: equal flight weights, NEES and 95 %
coverage on held-out flights). The vertical (rangefinder) variance is not
touched.

Input: flight directories of ``gz_map_companion_flight.py`` (calibrate or
test mode): ``companion_vo.csv`` (body-FRD velocity and model variances,
NED velocity) and the flight report with the ULog path. Reference: Gazebo
ground-truth velocity rotated into the true body axes. Samples the
companion's ``VelocityJumpGate`` rejects are left out (they never reach
EKF2), and only the flight segment of the report is used.

The first --fit-flights directories fit, the rest are held out. Nothing is
fitted on the held-out flights. Simulated flights: not a sensor
certification.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.companion_vo import VelocityJumpGate  # noqa: E402
from src.velocity_covariance_calibration import VelocityErrorFlight, calibrate_velocity_covariance  # noqa: E402


def _report(flight_dir: Path) -> dict:
    for name in ("gz_map_companion_calibrate.json", "gz_map_companion_test.json", "gz_vo_companion_flight.json"):
        if (flight_dir / name).exists():
            return json.loads((flight_dir / name).read_text())
    raise FileNotFoundError(f"no flight report in {flight_dir}")


def gate_mask(t_us: np.ndarray, vn: np.ndarray, ve: np.ndarray) -> np.ndarray:
    """True where the companion's jump gate accepts the sample."""
    gate = VelocityJumpGate()
    return np.array([gate.check(int(t), (a, b)) for t, a, b in zip(t_us, vn, ve)], dtype=bool)


def body_errors(flight_dir: Path):
    """(errors body x/y, model variances x/y, rejected-by-gate count)."""
    from pyulog import ULog
    rep = _report(flight_dir)
    offset = int((rep.get("companion_final_status") or {}).get("time_offset_us") or 0)
    rows = np.genfromtxt(flight_dir / "companion_vo.csv", delimiter=",", names=True)
    t = rows["t_us"] + offset
    ok = gate_mask(t, rows["vn"], rows["ve"])
    seg = (rep.get("segments") or {}).get("flight")
    t0, t1 = seg if seg else (rep.get("takeoff_us", 0) + 20e6, t[-1])
    var_all = np.column_stack([rows["var_x"], rows["var_y"]])
    sel = ok & (t >= t0) & (t <= t1) & np.all(var_all > 0, axis=1)
    u = ULog(rep["ulog"], ["vehicle_local_position_groundtruth", "vehicle_attitude_groundtruth"])
    d = {x.name: x.data for x in u.data_list}
    g, a = d["vehicle_local_position_groundtruth"], d["vehicle_attitude_groundtruth"]
    ts = t[sel]
    v_ned = np.column_stack([np.interp(ts, g["timestamp"], g[k]) for k in ("vx", "vy", "vz")])
    q = np.column_stack([np.interp(ts, a["timestamp"], a[f"q[{i}]"]) for i in range(4)])
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    r = np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
                  np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
                  np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], -2)
    truth_b = np.einsum("nji,nj->ni", r, v_ned)[:, :2]       # R^T v, body x/y
    v_b = np.column_stack([rows["vx"][sel], rows["vy"][sel]])
    return v_b - truth_b, var_all[sel], int((~ok).sum())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("flights", type=Path, nargs="+", help="fit flights first, then held-out flights")
    ap.add_argument("--fit-flights", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    if not 2 <= args.fit_flights <= len(args.flights) - 2:
        ap.error("need at least 2 fit and 2 held-out flights")
    data = [body_errors(f) for f in args.flights]
    flights = [VelocityErrorFlight(f"{f.parent.name}/{f.name}", tuple(map(tuple, e)),
                                   tuple(((float(v[0]), 0.0), (0.0, float(v[1]))) for v in var))
               for f, (e, var, _) in zip(args.flights, data)]
    cal = calibrate_velocity_covariance(flights[:args.fit_flights], flights[args.fit_flights:])
    out = {"model": "variance = scale * camera-VO model variance (horizontal body axes; vertical untouched)",
           "horizontal": {"scale": cal.scale, "bias_m_s": list(cal.bias_m_s),
                          "fit_mean_nees_2d": cal.fit_mean_nees_2d,
                          "holdout_mean_nees_2d": cal.holdout_mean_nees_2d,
                          "holdout_coverage_95pct": cal.holdout_coverage_95pct,
                          "holdout_flight_details": cal.holdout_flight_details,
                          "consistent_on_holdout": cal.consistent_on_holdout},
           "vertical": {"scale": 1.0},
           "fit_flights": list(cal.fit_flights), "holdout_flights": list(cal.holdout_flights),
           "jump_gate_rejected_per_flight": {fl.name: n for fl, (_, _, n) in zip(flights, data)},
           "samples_per_flight": {fl.name: len(e) for fl, (e, _, _) in zip(flights, data)},
           "note": "simulated flights (Gazebo, 3D swiss scene, 50 m); not a sensor certification"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("horizontal", "fit_flights")}, indent=1)[:2000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
