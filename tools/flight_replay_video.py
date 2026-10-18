#!/usr/bin/env python3
"""Replay video of a companion map flight (gz_map_companion_flight.py output).

Gazebo runs headless, so there is no screen recording. The video is drawn
from what the flight logged:

- on the onboard map (orthophoto, top view): true path (white), EKF2 estimate
  (cyan), map fixes - fused (green), rejected by NIS/step (red), during the
  map fault the shifted fixes as the companion saw them;
- the latest saved camera frame (every --save-every-th frame, a few seconds
  apart: not a video stream);
- phase (takeoff, GNSS on/off, route, map fault), estimate error and its
  history.

Usage:
    python tools/flight_replay_video.py FLIGHT_DIR --swiss-dir SCENE_DIR --out replay.mp4

Needs pyulog, OpenCV and ffmpeg (libx264). Not a navigation component.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

W, H = 1280, 720
MAP_PX = 720
FUSED, REJECTED = "FIX_FUSED", ("FIX_REJECTED_NIS", "FIX_REJECTED_STEP")
# BGR
C_TRUTH, C_EST, C_FUSED, C_REJ, C_TEXT = (255, 255, 255), (255, 220, 0), (80, 220, 80), (60, 60, 255), (235, 235, 235)
C_OPEN = (0, 200, 255)  # accepted by the localizer, not fused (calibration flights)


def ne_to_px(ne, georef: dict) -> np.ndarray:
    """Local N/E [m] -> map pixel (u, v); inverse of the georef affine."""
    a = np.asarray(georef["matrix_ne_m_per_px"], float)
    o = np.asarray(georef["origin_ne_m"], float)
    return (np.atleast_2d(np.asarray(ne, float)) - o) @ np.linalg.inv(a).T


def view_box(points_ne, *, margin_m: float = 15.0, min_size_m: float = 60.0):
    """Square N/E box (n_min, n_max, e_min, e_max) around the points."""
    p = np.asarray(points_ne, float)
    lo, hi = p.min(axis=0) - margin_m, p.max(axis=0) + margin_m
    c, half = (lo + hi) / 2, max(min_size_m, float(np.max(hi - lo))) / 2
    return c[0] - half, c[0] + half, c[1] - half, c[1] + half


def phase_at(t_us: int, report: dict) -> str:
    """Flight phase from the report's event times."""
    ev = report
    if "CALIBRATE" in str(ev.get("type", "")):
        return "calibration walk, GNSS on (fixes logged, not fused)" if t_us >= ev.get("takeoff_us", 0) \
            else "pre-flight"
    if ev.get("fault_start_us") and ev["fault_start_us"] <= t_us < ev.get("fault_end_us", 0):
        return "MAP FAULT: fixes shifted 3 m E"
    if ev.get("route_end_us") and t_us >= ev["route_end_us"]:
        return "hold / land (no GNSS)"
    if ev.get("gnss_off_us") and t_us >= ev["gnss_off_us"]:
        return "route, GNSS OFF: EKF2 on map + VO"
    if ev.get("takeoff_us") and t_us >= ev["takeoff_us"]:
        return "GNSS on, map lock"
    return "pre-flight"


def _report(flight: Path) -> dict:
    for name in ("gz_map_companion_test.json", "gz_map_companion_calibrate.json"):
        if (flight / name).exists():
            return json.loads((flight / name).read_text())
    raise FileNotFoundError(f"no gz_map_companion_*.json in {flight}")


def _fixes(flight: Path, offset_us: int) -> list[dict]:
    out = []
    with (flight / "companion_map.csv").open() as f:
        for r in csv.DictReader(f):
            if not r["localizer_status"].endswith("ACCEPTED_UNVALIDATED"):
                continue
            out.append({"t": int(r["t_us"]) + offset_us, "ne": (float(r["north_m"]), float(r["east_m"])),
                        "result": r["anchor_result"],
                        "shifted": float(r["injected_shift_n"]) != 0.0 or float(r["injected_shift_e"]) != 0.0})
    return out


def render(flight: Path, swiss_dir: Path, out: Path, *, speed: float = 4.0, fps: int = 10,
           map_image: str = "orthophoto.png") -> dict:
    from pyulog import ULog
    rep = _report(flight)
    u = ULog(rep["ulog"], ["vehicle_local_position", "vehicle_local_position_groundtruth"])
    d = {x.name: x.data for x in u.data_list}
    gt, est = d["vehicle_local_position_groundtruth"], d["vehicle_local_position"]
    t_gt = gt["timestamp"]
    p_gt = np.column_stack([gt["x"], gt["y"]])
    p_est = np.column_stack([np.interp(t_gt, est["timestamp"], est["x"]),
                             np.interp(t_gt, est["timestamp"], est["y"])])
    alt = -np.asarray(gt["z"], float)
    offset = int((rep.get("companion_final_status") or {}).get("time_offset_us") or 0)
    fixes = _fixes(flight, offset)
    frames = sorted((flight / "frames").glob("*.png"), key=lambda p: int(p.stem)) \
        if (flight / "frames").is_dir() else []
    frame_t = np.array([int(p.stem) + offset for p in frames])

    georef = json.loads((swiss_dir / "orthophoto_georef.json").read_text())
    ortho = cv2.imread(str(swiss_dir / map_image))
    t0 = rep.get("takeoff_us") or int(t_gt[0])
    flying = (t_gt >= t0) & (alt > 2.0)
    n0, n1, e0, e1 = view_box(p_gt[flying] if flying.any() else p_gt)
    corners = ne_to_px([(n1, e0), (n0, e1)], georef)          # top-left, bottom-right
    (u0, v0), (u1, v1) = corners.min(axis=0), corners.max(axis=0)
    crop = ortho[max(0, int(v0)):int(v1), max(0, int(u0)):int(u1)]
    base = cv2.resize(crop, (MAP_PX, MAP_PX), interpolation=cv2.INTER_AREA)
    base = cv2.addWeighted(base, 0.75, np.zeros_like(base), 0.25, 0)
    k = MAP_PX / (u1 - u0)

    def to_screen(ne):
        px = ne_to_px(ne, georef)
        return np.column_stack([(px[:, 0] - u0) * k, (px[:, 1] - v0) * k]).astype(np.int32)

    s_gt, s_est = to_screen(p_gt), to_screen(p_est)
    k0 = int(np.searchsorted(t_gt, t0))   # paths from takeoff: the estimator restarts on the ground
    err = np.linalg.norm(p_est - p_gt, axis=1)
    no_gnss = t_gt >= (rep.get("gnss_off_us") or 10 ** 18)
    t_end = int(t_gt[-1])
    times = np.arange(t0, t_end, speed * 1e6 / fps)

    out.parent.mkdir(parents=True, exist_ok=True)
    enc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                            "-s", f"{W}x{H}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
                            "-crf", "26", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
                           stdin=subprocess.PIPE)
    m_per_px = (n1 - n0) / MAP_PX
    for t in times:
        i = int(np.searchsorted(t_gt, t, side="right")) - 1
        img = np.zeros((H, W, 3), np.uint8)
        img[:, :MAP_PX] = base
        view = img[:, :MAP_PX]
        if i > k0:
            cv2.polylines(view, [s_gt[k0: i + 1].reshape(-1, 1, 2)], False, C_TRUTH, 2, cv2.LINE_AA)
            cv2.polylines(view, [s_est[k0: i + 1].reshape(-1, 1, 2)], False, C_EST, 1, cv2.LINE_AA)
        for f in fixes:
            if f["t"] > t:
                break
            (x, y), = to_screen([f["ne"]])
            if f["result"] == FUSED:
                cv2.circle(view, (int(x), int(y)), 2, C_FUSED, -1, cv2.LINE_AA)
            elif f["result"] in REJECTED:
                cv2.drawMarker(view, (int(x), int(y)), C_REJ, cv2.MARKER_TILTED_CROSS, 7, 1, cv2.LINE_AA)
            elif not f["result"]:
                cv2.circle(view, (int(x), int(y)), 2, C_OPEN, -1, cv2.LINE_AA)
        cv2.circle(view, tuple(int(c) for c in s_gt[i]), 7, C_TRUTH, 2, cv2.LINE_AA)
        cv2.circle(view, tuple(int(c) for c in s_est[i]), 5, C_EST, -1, cv2.LINE_AA)
        bar = int(round(10.0 / m_per_px))
        cv2.line(view, (20, H - 25), (20 + bar, H - 25), C_TEXT, 3)
        cv2.putText(view, "10 m", (20, H - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_TEXT, 1, cv2.LINE_AA)
        cv2.putText(view, "N", (MAP_PX - 30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, C_TEXT, 2, cv2.LINE_AA)
        cv2.arrowedLine(view, (MAP_PX - 24, 70), (MAP_PX - 24, 40), C_TEXT, 2, tipLength=0.4)

        x0 = MAP_PX + 20
        j = int(np.searchsorted(frame_t, t, side="right")) - 1
        if j >= 0:
            cam = cv2.imread(str(frames[j]))
            cam = cv2.resize(cam, (480, 360), interpolation=cv2.INTER_NEAREST)
            img[20:380, x0:x0 + 480] = cam
            cv2.putText(img, f"camera frame t={frame_t[j] / 1e6:.0f} s (saved every few s)", (x0, 398),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_TEXT, 1, cv2.LINE_AA)
        n_fused = sum(f["t"] <= t and f["result"] == FUSED for f in fixes)
        n_rej = sum(f["t"] <= t and f["result"] in REJECTED for f in fixes)
        n_open = sum(f["t"] <= t and not f["result"] for f in fixes)
        lines = [(f"t = {t / 1e6:6.1f} s   alt {alt[i]:5.1f} m", C_TEXT),
                 (phase_at(int(t), rep), C_REJ if "FAULT" in phase_at(int(t), rep) else C_TEXT),
                 (f"EKF2 error {err[i]:.2f} m" + ("   (GNSS off)" if no_gnss[i] else ""), C_EST),
                 (f"map fixes: fused {n_fused}  rejected {n_rej}" + (f"  logged {n_open}" if n_open else ""),
                  C_FUSED)]
        for r, (txt, col) in enumerate(lines):
            cv2.putText(img, txt, (x0, 430 + 26 * r), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 1, cv2.LINE_AA)
        legend = [("truth", C_TRUTH), ("EKF2", C_EST), ("fused", C_FUSED), ("rejected", C_REJ), ("logged", C_OPEN)]
        for r, (txt, col) in enumerate(legend):
            cv2.circle(img, (x0 + 8 + 105 * r, 548), 5, col, -1)
            cv2.putText(img, txt, (x0 + 18 + 105 * r, 553), cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_TEXT, 1, cv2.LINE_AA)
        # Error history strip, 0..3 m.
        gx0, gy0, gw, gh = x0, 575, W - x0 - 20, 120
        cv2.rectangle(img, (gx0, gy0), (gx0 + gw, gy0 + gh), (90, 90, 90), 1)
        sel = slice(int(np.searchsorted(t_gt, t0)), i + 1)
        if i > 0:
            tx = gx0 + (t_gt[sel] - t0) / (t_end - t0) * gw
            ty = gy0 + gh - np.clip(err[sel] / 3.0, 0, 1) * gh
            cv2.polylines(img, [np.column_stack([tx, ty]).astype(np.int32).reshape(-1, 1, 2)], False, C_EST, 1)
        for key, col in (("gnss_off_us", C_TEXT), ("fault_start_us", C_REJ), ("fault_end_us", C_REJ)):
            if rep.get(key):
                xx = int(gx0 + (rep[key] - t0) / (t_end - t0) * gw)
                cv2.line(img, (xx, gy0), (xx, gy0 + gh), col, 1)
        cv2.putText(img, "EKF2 error, 0-3 m", (gx0 + 4, gy0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_TEXT, 1)
        enc.stdin.write(img.tobytes())
    enc.stdin.close()
    if enc.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    return {"video": str(out), "frames": len(times), "seconds": round(len(times) / fps, 1),
            "camera_frames": len(frames), "fixes": len(fixes)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("flight", type=Path, help="flight output directory")
    ap.add_argument("--swiss-dir", type=Path, required=True, help="scene directory (orthophoto + georef)")
    ap.add_argument("--map-image", default="orthophoto.png")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--speed", type=float, default=4.0, help="sim seconds per video second")
    ap.add_argument("--fps", type=int, default=10)
    a = ap.parse_args(argv)
    print(json.dumps(render(a.flight, a.swiss_dir, a.out, speed=a.speed, fps=a.fps, map_image=a.map_image)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
