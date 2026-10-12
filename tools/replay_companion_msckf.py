#!/usr/bin/env python3
"""Offline replay of a companion MSCKF recording (``--msckf-record``).

Feeds the recorded autopilot IMU samples and per-frame inputs (KLT
features, AHRS attitude, rangefinder) to ``CompanionMsckf`` in time order,
as the companion does, and reports the counters, the processing time per
frame and the body-FRD velocity against the autopilot's LOCAL_POSITION_NED
velocity (GNSS-aided while GNSS is on: a reference, not ground truth).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.camera_rotation_compensation import body_to_ned  # noqa: E402
from src.companion_msckf import CompanionMsckf  # noqa: E402
from src.imu_preintegration import ImuReading  # noqa: E402
from src.msckf_vio import PinholeCamera  # noqa: E402
from tools.gz_vo_companion import BODY_FROM_CAMERA, MSCKF_IMU_NOISE, calibration  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("record", type=Path)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--hfov", type=float, default=1.2)
    ap.add_argument("--camera-offset-frd", type=float, nargs=3, default=(0.0, 0.0, 0.1))
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    imu, frames, lpos = [], [], []
    for line in args.record.read_text().splitlines():
        try:
            d = json.loads(line)
        except ValueError:
            continue          # truncated last line of a killed run
        if "imu" in d:
            imu.append(ImuReading(int(d["imu"]), tuple(d["g"]), tuple(d["a"])))
        elif "frame" in d:
            frames.append(d)
        elif "lpos" in d:
            lpos.append((int(d["lpos"]), d["v"]))
    imu.sort(key=lambda r: r.timestamp_us)
    gaps = np.diff([r.timestamp_us for r in imu]) * 1e-3 if len(imu) > 1 else np.zeros(1)
    cal = calibration(args.width, args.height, args.hfov, 0.0)
    m = CompanionMsckf(PinholeCamera(cal.focal_x_px, cal.focal_y_px, cal.center_x_px, cal.center_y_px,
                                     BODY_FROM_CAMERA, tuple(args.camera_offset_frd), pixel_sigma=1.0),
                       MSCKF_IMU_NOISE)
    lt = np.array([t for t, _ in lpos]) if lpos else np.zeros(0)
    i, rows, ms, errors = 0, [], [], {}
    for f in frames:
        t = int(f["frame"])
        while i < len(imu) and imu[i].timestamp_us <= t + 20_000:
            m.add_imu(imu[i])
            i += 1
        c0 = time.perf_counter()
        out = m.process(t, {int(k): tuple(v) for k, v in f["features"].items()}, attitude_q=tuple(f["q"]),
                        roll=f["roll"], pitch=f["pitch"], range_m=f["range"])
        ms.append((time.perf_counter() - c0) * 1e3)
        if m.last_error:
            errors[m.last_error] = errors.get(m.last_error, 0) + 1
            m.last_error = None
        if out is None or not len(lt):
            continue
        j = int(np.searchsorted(lt, t))
        if not 0 < j < len(lt) or lt[j] - lt[j - 1] > 200_000:
            continue
        v_ref = np.asarray(lpos[j][1] if lt[j] - t < t - lt[j - 1] else lpos[j - 1][1])
        v_b_ref = body_to_ned(f["roll"], f["pitch"], f["yaw"]).T @ v_ref
        rows.append((t, *out.velocity_frd, *v_b_ref, *out.variance_frd))
    report = {"frames": len(frames), "imu_samples": len(imu),
              "imu_gap_ms": {"median": float(np.median(gaps)), "max": float(np.max(gaps)),
                             "over_50ms": int(np.sum(gaps > 50))},
              "counts": m.counts, "errors": errors,
              "process_ms": {"median": float(np.median(ms)), "p95": float(np.percentile(ms, 95)),
                             "max": float(np.max(ms))} if ms else None}
    if rows:
        a = np.asarray(rows)
        e = a[:, 1:4] - a[:, 4:7]
        nees = np.mean(np.sum(e[:, :2] ** 2 / a[:, 7:9], axis=1))
        report["velocity_vs_autopilot"] = {
            "samples": len(a), "rmse_xy_m_s": float(math.sqrt(np.mean(np.sum(e[:, :2] ** 2, axis=1)))),
            "bias_frd_m_s": [float(x) for x in e.mean(axis=0)], "mean_nees_xy": float(nees)}
    print(json.dumps(report, indent=1))
    if args.out:
        args.out.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
