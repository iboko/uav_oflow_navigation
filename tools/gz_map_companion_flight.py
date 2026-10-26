#!/usr/bin/env python3
"""Orthophoto map correction on the companion computer -> EKF2: PX4 SITL + Gazebo.

The companion (tools/gz_vo_companion.py) runs VO and, at 2 Hz, the
orthophoto localizer (src/orthophoto_localization.py) on the same camera.
The orthophoto is the scene texture DEGRADED like a survey product: half
the resolution, blurred, different contrast and noise, optionally a
georeferencing error. A VO scale error (focal length) is injected in both
modes: the map has to correct it.

  --mode calibrate   one short GNSS flight; map fixes are logged, never
                     sent. Errors against Gazebo ground truth (from the ulog,
                     after the flight) -> <out>/map_errors.json. Run it for
                     five separate flights: the fix error depends on the
                     EKF2 tilt error of that flight (1-2 deg in SITL, i.e.
                     0.1-0.2 m at 5 m), constant within one flight
  --mode combine     no simulation: calibrate_map_error over the five
                     flights (3 fit + 2 held out) -> map_calibration.json
  --mode test        uses that calibration (fuse mode):
     1  map lock with GNSS, EKF2 fuses EV position (cs_ev_pos)
     2  GNSS lost in flight (EKF2_GPS_CTRL=0): EKF2 on map + VO only
     3  open "staircase" route (4 legs, not closed, so a VO scale error
        accumulates): absolute horizontal error EKF2 vs truth; the VO-only
        track of the same flight is reported for comparison
     4  map fault: fixes shifted 3 m (wrong georeference / false match);
        none may be fused (from the companion map log), the vehicle must
        not move; after the fault the fixes are fused again
     5  AUTO.LAND

Simulated flights with an idealised camera: the calibration stays a
simulation result (validated_for_flight=false).
"""
from __future__ import annotations

import argparse
import json
import math
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from pymavlink import mavutil

import gz_failure_scenarios as gz
import gz_vo_companion_flight as vo_flight
from gz_no_gnss_flight import topic, xy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.map_error_field import cross_validated_field_nees, field_nees, fit_field  # noqa: E402
from src.map_error_calibration import (MapErrorFlight, calibrate_map_error, cross_validated_nees,  # noqa: E402
                                       fit_quality_model, place_outlier_rule, place_weighted_error_model,
                                       quality_covariance)

# EV horizontal position + velocity. 5 m: at 3 m the camera sees the ground
# at 1.6x the map resolution and ORB matching fails (checked offline).
PARAMS = dict(vo_flight.PARAMS, EKF2_EV_CTRL="5", MIS_TAKEOFF_ALT="5.0")
TAKEOFF_M = 5.0
FALSE_FIX_M = 1.0


def make_orthophoto(px4_dir: Path, out: Path, georef_error_ne_m=(0.0, 0.0), seed: int = 3) -> tuple[Path, Path]:
    """Survey-like map of the ofnav_textured ground plane (20 x 20 m, centred).

    Texture orientation on the Gazebo plane: column -> +x (east), row -> -y
    (south). A georeferencing error shifts the map coordinates.
    """
    import cv2
    src = cv2.imread(str(px4_dir / "Tools" / "simulation" / "gz" / "worlds" / f"{gz.TEXTURED_WORLD}.png"),
                     cv2.IMREAD_GRAYSCALE)
    if src is None:
        raise RuntimeError("textured world image missing")
    rng = np.random.default_rng(seed)
    img = cv2.resize(src, (src.shape[1] // 2, src.shape[0] // 2), interpolation=cv2.INTER_AREA)
    img = cv2.GaussianBlur(img, (0, 0), 1.0).astype(np.float32)
    img = 255.0 * (img / 255.0) ** 0.8 * 0.9 + 12.0 + rng.normal(0, 4.0, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    s = 20.0 / img.shape[1]
    georef = {"origin_ne_m": [10.0 - 0.5 * s + georef_error_ne_m[0], -10.0 + 0.5 * s + georef_error_ne_m[1]],
              "matrix_ne_m_per_px": [[0.0, -s], [s, 0.0]],
              "note": "pixel (u,v) -> NE; world ENU x=east, y=north; injected georef error "
                      f"{list(georef_error_ne_m)} m"}
    map_png, georef_json = out / "orthophoto.png", out / "orthophoto_georef.json"
    cv2.imwrite(str(map_png), img)
    georef_json.write_text(json.dumps(georef, indent=2))
    return map_png, georef_json


def latest_status(log: Path) -> dict:
    lines = [ln for ln in log.read_text().splitlines() if ln.startswith('{"sim_s"')]
    return json.loads(lines[-1]) if lines else {}


# Map fault (check 4): the vehicle must be still before the shift, else the
# true drift picks up the settling after the last route leg (test FB2 diag:
# 0.22 m/s at the fault start, 0.61 m "drift" of which ~0.3 m was settling).
SETTLE_MAX_SPEED_MPS = 0.1
SETTLE_HOLD_S = 3.0
SETTLE_TIMEOUT_S = 30.0
DRIFT_MEAN_S = 2.0


def wait_settled(read_ne, sleep, *, step_s: float = 0.5, max_speed: float = SETTLE_MAX_SPEED_MPS,
                 hold_s: float = SETTLE_HOLD_S, timeout_s: float = SETTLE_TIMEOUT_S):
    """Sim seconds until the speed from successive positions stayed below
    max_speed for hold_s; None if not within timeout_s."""
    prev, still, waited = np.asarray(read_ne(), float), 0.0, 0.0
    while waited < timeout_s:
        sleep(step_s)
        waited += step_s
        cur = np.asarray(read_ne(), float)
        still = still + step_s if np.linalg.norm(cur - prev) / step_s < max_speed else 0.0
        prev = cur
        if still >= hold_s:
            return waited
    return None


def mean_position(read_ne, sleep, *, seconds: float = DRIFT_MEAN_S, step_s: float = 0.5) -> np.ndarray:
    """Mean of positions read over seconds (hover oscillation averaged out)."""
    pts = [np.asarray(read_ne(), float)]
    for _ in range(int(round(seconds / step_s))):
        sleep(step_s)
        pts.append(np.asarray(read_ne(), float))
    return np.mean(pts, axis=0)


def ulog_truth(px4_dir: Path, since_wall: float):
    from pyulog import ULog
    logs = sorted((px4_dir / "build" / "px4_sitl_default" / "rootfs" / "log").glob("*/*.ulg"),
                  key=lambda p: p.stat().st_mtime)
    logs = [p for p in logs if p.stat().st_mtime >= since_wall]
    if not logs:
        raise RuntimeError("no ulog from this flight")
    u = ULog(str(logs[-1]), ["vehicle_local_position_groundtruth", "vehicle_local_position",
                             "vehicle_attitude", "vehicle_attitude_groundtruth"])
    d = {x.name: x.data for x in u.data_list}
    return d, str(logs[-1])


def yaw_deg(att) -> np.ndarray:
    return np.degrees(np.arctan2(2 * (att["q[0]"] * att["q[3]"] + att["q[1]"] * att["q[2]"]),
                                 1 - 2 * (att["q[2]"] ** 2 + att["q[3]"] ** 2)))


def heading_error_deg(d: dict, t0_us: int, gnss_off_us: int | None = None, min_alt_m: float = 2.0) -> dict:
    """EKF2 heading minus ground truth while airborne (diagnostic only).

    Airborne: after t0_us and more than min_alt_m above the start (true
    altitude). Before the map lock the heading is the magnetometer's; on the
    ground after landing the true yaw jumps at touchdown. Both used to set the
    maximum (8-11 deg) of flights whose in-flight error stayed below 1.5 deg."""
    a, g = d["vehicle_attitude"], d["vehicle_attitude_groundtruth"]
    gt = d["vehicle_local_position_groundtruth"]
    t = a["timestamp"]
    alt = -np.interp(t, gt["timestamp"], gt["z"])
    sel = (t >= t0_us) & (alt > min_alt_m)
    if not sel.any():
        return {"max_abs_deg": None, "mean_deg": None}
    truth = np.interp(t, g["timestamp"], np.unwrap(np.radians(yaw_deg(g))))
    err = np.degrees((np.radians(yaw_deg(a)) - truth + np.pi) % (2 * np.pi) - np.pi)
    out = {"max_abs_deg": round(float(np.max(np.abs(err[sel]))), 2), "mean_deg": round(float(np.mean(err[sel])), 2),
           "airborne_from_alt_m": min_alt_m}
    if gnss_off_us is not None and (sel & (t >= gnss_off_us)).any():
        out["no_gnss_max_abs_deg"] = round(float(np.max(np.abs(err[sel & (t >= gnss_off_us)]))), 2)
    return out


def map_heading_errors_deg(d: dict, rows: list[dict], t_us: np.ndarray) -> np.ndarray:
    """(EKF2 heading + map heading offset) minus true heading at each fix [deg].

    The offset is fix heading minus the autopilot heading used for that
    frame; NaN where the NCC search gave no heading."""
    a, g = d["vehicle_attitude"], d["vehicle_attitude_groundtruth"]
    est = np.interp(t_us, a["timestamp"], np.unwrap(np.radians(yaw_deg(a))))
    truth = np.interp(t_us, g["timestamp"], np.unwrap(np.radians(yaw_deg(g))))
    off = np.array([float(r["heading_offset_rad"]) for r in rows])
    return np.degrees((est + off - truth + np.pi) % (2 * np.pi) - np.pi)


def interp(data, t_us: np.ndarray, keys=("x", "y")) -> np.ndarray:
    return np.column_stack([np.interp(t_us, data["timestamp"], data[k]) for k in keys])


def tilt_error_deg(d: dict, t0_us: int) -> dict:
    """EKF2 roll/pitch minus ground truth, mean over the flight (diagnostic)."""
    def rp(a):
        w, x, y, z = a["q[0]"], a["q[1]"], a["q[2]"], a["q[3]"]
        return (np.degrees(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))),
                np.degrees(np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))))
    a, g = d["vehicle_attitude"], d["vehicle_attitude_groundtruth"]
    sel = a["timestamp"] >= t0_us
    (ra, pa), (rg, pg) = rp(a), rp(g)
    return {"roll": round(float(np.mean(ra[sel] - np.interp(a["timestamp"][sel], g["timestamp"], rg))), 2),
            "pitch": round(float(np.mean(pa[sel] - np.interp(a["timestamp"][sel], g["timestamp"], pg))), 2)}


def heading_phase(args, px4: gz.Px4, link, report: dict, check) -> None:
    """After GNSS loss: EKF2 heading vs truth with a magnetic heading error from boot."""
    report["mag_decl_error"] = {"wmm_declination_deg": args.wmm_declination_deg,
                                "error_deg": args.mag_decl_error_deg, "set_at": "boot"}
    samples = []
    for _ in range(20):
        gz.sim_sleep(2)
        est = float(topic(px4, "vehicle_local_position").get("heading", math.nan))
        truth = float(topic(px4, "vehicle_local_position_groundtruth").get("heading", math.nan))
        f = topic(px4, "estimator_status_flags")
        err = math.degrees((est - truth + math.pi) % (2 * math.pi) - math.pi)
        samples.append({"t": round(link.boot_s, 1), "heading_error_deg": round(err, 2),
                        "cs_ev_yaw": f.get("cs_ev_yaw"), "cs_mag_hdg": f.get("cs_mag_hdg"),
                        "cs_mag_3d": f.get("cs_mag_3d")})
        print(samples[-1], flush=True)
    report["heading_samples"] = samples
    late = [abs(x["heading_error_deg"]) for x in samples[10:]]
    info = dict(max_abs_heading_error_last_20s_deg=round(max(late), 2),
                ev_yaw_fused=samples[-1]["cs_ev_yaw"], mag_fused=samples[-1]["cs_mag_hdg"] == "True"
                or samples[-1]["cs_mag_3d"] == "True")
    if args.ev_yaw:
        check("H_heading_held_by_map_despite_mag_error", max(late) < 4.0 and samples[-1]["cs_ev_yaw"] == "True",
              **info)
    else:
        report["heading_without_ev_yaw"] = info  # reference run, not a pass criterion
        print("INFO heading_without_ev_yaw", info, flush=True)


def heading_calibration(dirs, nf: int = 3) -> dict | None:
    """Map heading error: mean and SD on the nf fit flights, coverage of
    mean +- 2 SD on the held-out flights (diagnostic; MapAnchor uses the
    spread of its own fixes, floored at 0.05 rad)."""
    per = []
    for d in dirs:
        f = json.loads((d / "map_errors.json").read_text())
        h = np.asarray([x for x in f.get("heading_errors_deg", []) if x is not None and math.isfinite(x)])
        per.append(h)
    if len(per) < 5 or any(len(h) < 10 for h in per):
        return None
    fit = np.concatenate(per[:nf])
    m, sd = float(np.mean(fit)), float(np.std(fit))
    hold = np.concatenate(per[nf:])
    return {"fit_mean_deg": round(m, 2), "fit_sd_deg": round(sd, 2),
            "per_flight_mean_deg": [round(float(np.mean(h)), 2) for h in per],
            "holdout_within_2sd": round(float(np.mean(np.abs(hold - m) <= 2 * sd)), 3)}


def source_calibration(dirs, nf: int = 3, raw=None) -> dict | None:
    """Per map source (primary / fallback): bias and isotropic covariance on
    the nf fit flights (equal weight), NEES and coverage on the held-out ones."""
    raw = raw if raw is not None else [json.loads((d / "map_errors.json").read_text()) for d in dirs]
    if not all("map_source" in r for r in raw) or len({s_ for r in raw for s_ in r["map_source"]}) < 2:
        return None
    out = {"bias": {}, "cov": {}, "check": {}}
    for src in sorted({s_ for r in raw for s_ in r["map_source"]}):
        per = [np.asarray(r["errors_ne_m"], float)[np.asarray(r["map_source"]) == src] for r in raw]
        fit = [e for e in per[:nf] if len(e) >= 10]
        if not fit:
            continue
        b = np.mean([e.mean(axis=0) for e in fit], axis=0)
        var = float(np.mean([np.mean(np.sum((e - b) ** 2, axis=1)) / 2 for e in fit]))
        nees = [np.sum((e - b) ** 2, axis=1) / var for e in per[nf:] if len(e)]
        out["bias"][src] = b.round(4).tolist()
        out["cov"][src] = [[var, 0.0], [0.0, var]]
        out["check"][src] = {"fit_fixes": [len(e) for e in per[:nf]], "holdout_fixes": [len(e) for e in per[nf:]],
                             "sigma_m": round(math.sqrt(var), 3),
                             "holdout_mean_nees": [round(float(np.mean(n)), 2) for n in nees],
                             "holdout_coverage_95": [round(float(np.mean(n <= 5.991)), 3) for n in nees]}
    return out


def quality_calibration(dirs, bias, nf: int = 3, raw=None) -> dict | None:
    """ML fit of R = floor*I + k2*C_subpatch on the nf fit flights, NEES on
    the held-out ones. Fixes without a sub-patch covariance are left out
    (counted)."""
    errs, subs, dropped = [], [], []
    raw = raw if raw is not None else [json.loads((d / "map_errors.json").read_text()) for d in dirs]
    for f in raw:
        if "subpatch_cov_ne_m2" not in f:
            return None
        e = np.asarray(f["errors_ne_m"], float)
        c = np.asarray(f["subpatch_cov_ne_m2"], float)
        ok = np.isfinite(c).all(axis=(1, 2))
        errs.append(e[ok]); subs.append(c[ok]); dropped.append(int((~ok).sum()))
    m = fit_quality_model(errs[:nf], subs[:nf], errs[nf:], subs[nf:], bias)
    m["fixes_without_subpatch"] = dropped
    return m


def place_diagnostics(dirs, raw, fit_flights: int = 3) -> dict | None:
    """Place-weighted model on the fit flights and the cross-validated
    holdout NEES of both estimators (diagnostic: covariance_ne_m2 stays the
    per-fix estimate). Only for calibrations over many places (walk)."""
    if not all(r.get("outlier_rule") == "place" for r in raw):
        return None
    errs = [np.asarray(r["errors_ne_m"]) for r in raw]
    pos = [flight_positions(d, r) for d, r in zip(dirs, raw)]
    model = place_weighted_error_model(errs[:fit_flights], pos[:fit_flights])
    b = np.asarray(model["bias_ne_m"])
    v = model["covariance_ne_m2"][0][0]
    nees = [np.sum((e - b) ** 2, axis=1) / v for e in errs[fit_flights:]]
    model["holdout_mean_nees"] = [round(float(np.mean(n)), 2) for n in nees]
    model["holdout_check"] = {"coverage": float(np.mean([np.mean(n <= 5.991464547107979) for n in nees])),
                              "mean_nees": float(np.mean([np.mean(n) for n in nees])),
                              "per_flight_mean_nees": model["holdout_mean_nees"]}
    model["cross_validation_3_fit"] = cross_validated_nees(errs, pos, fit_flights=3)
    if len(errs) > 5:
        # More places in the fit: does the holdout consistency improve?
        model[f"cross_validation_{len(errs) - 2}_fit"] = cross_validated_nees(errs, pos, fit_flights=len(errs) - 2)
    return model


# The test routes (forward = east, right = south at the take-off heading)
# and their start lie here, in local N/E: calibration fixes inside are not
# used (combine), and the calibration walk ends a leg on entering it.
TEST_ZONE = "north < 10 m and east > -10 m"
# N and E ranges (m) covering all test routes: where the field variance is
# taken for the route limit.
TEST_ROUTE_AREA = ((-60.0, 15.0), (-10.0, 65.0))
TEST_ROUTES = {
    # Legs of the test box as (pitch, roll) stick fractions; all stay in TEST_ZONE.
    "box": ((1, 0), (0, 1), (1, 0), (0, 1)),
    "box_rf": ((0, 1), (1, 0), (0, 1), (1, 0)),
    "diag": ((0.7, 0.7), (1, 0), (0.7, 0.7), (0, 1)),
}


# Area over the test routes (their fixes span N -57..0 m, E 0..60 m) for
# --calib-walk-in-test-zone: (north range, east range), m.
CALIB_ZONE_BOX = ((-75.0, 10.0), (-10.0, 75.0))


def in_test_zone(north_m, east_m):
    return (np.asarray(north_m) < 10.0) & (np.asarray(east_m) > -10.0)


def flight_positions(d: Path, f: dict) -> np.ndarray:
    """True positions of the inlier fixes of a calibration flight.

    Stored in map_errors.json since series M; for older flights recomputed
    from the flight's ulog and map log with the same place rule, and
    checked against the stored errors."""
    if "positions_ne_m" in f:
        return np.asarray(f["positions_ne_m"], dtype=float)
    from pyulog import ULog
    rep = json.loads((d / "gz_map_companion_calibrate.json").read_text())
    truth = ULog(rep["ulog"], ["vehicle_local_position_groundtruth"]).data_list[0].data
    rows = read_map_log(d / "companion_map.csv")
    offset = int(rep["companion_final_status"].get("time_offset_us") or 0)
    acc = [r for r in rows if r["localizer_status"].endswith("ACCEPTED_UNVALIDATED")]
    t = np.array([int(r["t_us"]) + offset for r in acc])
    fx = np.array([[float(r["north_m"]) - float(r["injected_shift_n"]),
                    float(r["east_m"]) - float(r["injected_shift_e"])] for r in acc])
    t0, t1 = rep["segments"]["flight"]
    sel = (t >= t0) & (t <= t1)
    pos = interp(truth, t)[sel]
    e = fx[sel] - pos
    inl = place_outlier_rule(e, pos)["inlier"]
    if not np.allclose(e[inl], np.asarray(f["errors_ne_m"]), atol=1e-3):
        raise RuntimeError(f"{d}: recomputed fixes differ from map_errors.json")
    return pos[inl]


def combine(args) -> None:
    nf = args.fit_flights
    flights, heights = [], []
    raw = [json.loads((d / "map_errors.json").read_text()) for d in args.flights]
    excluded = None
    if getattr(args, "exclude_test_zone", False):
        # Calibration independent of the test area: drop the fixes (and
        # their per-fix fields) inside TEST_ZONE.
        excluded = []
        for d, f in zip(args.flights, raw):
            pos = flight_positions(d, f)
            keep = ~in_test_zone(pos[:, 0], pos[:, 1])
            excluded.append(int((~keep).sum()))
            for k in ("errors_ne_m", "map_source", "subpatch_cov_ne_m2"):
                if k in f:
                    f[k] = [x for x, m in zip(f[k], keep) if m]
            f["positions_ne_m"] = pos[keep].tolist()
        # Flights with fewer than 30 fixes left are dropped (the
        # calibration's own minimum), order kept; fit = the first nf.
        kept = [i for i, f in enumerate(raw) if len(f["errors_ne_m"]) >= 30]
        dropped_flights = [raw[i]["name"] for i in range(len(raw)) if i not in kept]
        args.flights = [args.flights[i] for i in kept]
        raw = [raw[i] for i in kept]
        excluded = {"fixes_per_flight": [excluded[i] for i in kept], "dropped_flights": dropped_flights}
    for f in raw:
        if "median_height_m" in f:
            heights.append(float(f["median_height_m"]))
        flights.append(MapErrorFlight(f["name"], tuple(map(tuple, f["errors_ne_m"])), f["total_frames"],
                                      f["false_fixes"]))
    heights = heights if len(heights) == len(args.flights) else []
    rule = raw[0].get("outlier_rule", f"|error - flight median| > {FALSE_FIX_M} m")
    if rule == "place":
        rule = "place rule: 20 m places, place > 3 m off = false place, fix > 1 m from place median = false fix"
    cal = calibrate_map_error(flights[:nf], flights[nf:], min_samples_per_flight=30,
                              outliers_removed_by_rule=rule)
    per_source = source_calibration(args.flights, nf, raw)
    if per_source:
        # Fixes pre-corrected by their source's bias in the companion: the
        # pooled calibration is redone on per-source-corrected errors.
        corrected = []
        for r, f in zip(raw, flights):
            b = np.array([per_source["bias"][s_] for s_ in r["map_source"]])
            corrected.append(MapErrorFlight(f.name, tuple(map(tuple, np.asarray(f.errors_ne_m) - b)),
                                            f.total_frames, f.false_fixes))
        flights = corrected
        cal = calibrate_map_error(flights[:nf], flights[nf:], min_samples_per_flight=30, outliers_removed_by_rule=rule)
    cov = np.asarray(cal.covariance_ne_m2)
    within = np.asarray(cal.within_flight_covariance_ne_m2)
    between = np.asarray(cal.between_flight_covariance_ne_m2)
    holdout = {"coverage": cal.holdout_nees_coverage_95pct, "mean_nees": cal.holdout_mean_nees_2d}
    if args.isotropic:
        cov = np.eye(2) * np.trace(cov) / 2
        within = np.eye(2) * np.trace(within) / 2
        between = np.eye(2) * np.trace(between) / 2
        nees = []
        for f in flights[nf:]:
            e = np.asarray(f.errors_ne_m) - np.asarray(cal.bias_ne_m)
            nees.append(np.einsum("ij,ij->i", e, e) / cov[0, 0])
        holdout = {"coverage": float(np.mean([np.mean(n <= 5.991464547107979) for n in nees])),
                   "mean_nees": float(np.mean([np.mean(n) for n in nees])),
                   "per_flight_mean_nees": [round(float(np.mean(n)), 2) for n in nees]}
    place_model = place_diagnostics(args.flights, raw, nf)
    bias = list(cal.bias_ne_m)
    model = "isotropic trace/2" if args.isotropic else "empirical 2x2"
    if args.place_weighted:
        # Bias + place offset + noise, one weight per place (see
        # place_weighted_error_model). No within/between-flight split: the
        # companion uses the total at every height.
        if place_model is None:
            raise SystemExit("--place-weighted needs flights with the place rule")
        bias, cov = place_model["bias_ne_m"], np.asarray(place_model["covariance_ne_m2"])
        within = between = None
        holdout = place_model["holdout_check"]
        model = "place-weighted isotropic"
    flight_scale = None
    if args.flight_quantile is not None:
        # Flight-level conservative R: the map error is regional, so a whole
        # held-out flight can sit at NEES 5 with the per-fix covariance. Scale
        # R so that the given quantile of held-out flights' mean NEES (cross
        # validation on the calibration flights only, n-2 fit) is at 2.
        # Fit flights only (n_fit - 2 per split): the held-out flights stay out.
        if place_model is None or nf < 5:
            raise SystemExit("--flight-quantile needs the place rule and at least 5 fit flights")
        cv = cross_validated_nees([np.asarray(r["errors_ne_m"]) for r in raw[:nf]],
                                  [flight_positions(d, r) for d, r in zip(args.flights[:nf], raw[:nf])],
                                  fit_flights=nf - 2)
        est = "place_weighted" if args.place_weighted else "per_fix"
        q = float(np.quantile(cv[est]["per_holdout"], args.flight_quantile))
        flight_scale = max(1.0, q / 2.0)
        cov = cov * flight_scale
        within = None if within is None else within * flight_scale
        between = None if between is None else between * flight_scale
        holdout = {"before_scale": holdout,
                   "per_flight_mean_nees": [round(x / flight_scale, 2) for x in holdout["per_flight_mean_nees"]]}
        model += f", flight-level scale {flight_scale:.2f} (q{args.flight_quantile:g} of CV flight NEES / 2)"
    field_out = None
    if getattr(args, "error_field", False):
        # Kriged offset field (src/map_error_field.py): bias and variance per
        # fix position in the companion. covariance_ne_m2 becomes the largest
        # field variance over the test-route area (TEST_ROUTE_AREA), fixed
        # before any test flight, so the route limit stays 3 sigma.
        errs = [np.asarray(r["errors_ne_m"]) for r in raw]
        pos = [flight_positions(d, r) for d, r in zip(args.flights, raw)]
        field = fit_field(errs[:nf], pos[:nf], drift=getattr(args, "field_drift", "constant"))
        nees = [float(np.mean(field_nees(field, e, p))) for e, p in zip(errs[nf:], pos[nf:])]
        gn, ge = np.meshgrid(np.arange(*TEST_ROUTE_AREA[0], 5.0), np.arange(*TEST_ROUTE_AREA[1], 5.0))
        area_var = float(field.fix_variance(np.column_stack([gn.ravel(), ge.ravel()])).max())
        field_out = {**field.to_json(),
                     "holdout_mean_nees": [round(x, 2) for x in nees],
                     "cross_validation": cross_validated_field_nees(errs, pos, fit_flights=len(errs) - 2,
                                                                 drift=getattr(args, "field_drift", "constant"))
                     if len(errs) > 4 else None,
                     "test_route_area_max_variance_m2": area_var}
        bias, cov = [0.0, 0.0], np.eye(2) * area_var
        within = between = None
        holdout = {"per_flight_mean_nees": field_out["holdout_mean_nees"]}
        model = "kriged error field (bias and variance per fix position)"
    out = {"bias_ne_m": bias, "covariance_ne_m2": cov.tolist(),
           "covariance_model": model,
           "holdout_check": holdout,
           "sigma_ne_m": [math.sqrt(cov[i, i]) for i in (0, 1)],
           # cov = within + between; the companion scales the between-flight
           # (tilt) part with (h / calibration_height_m)^2 per fix.
           "within_flight_covariance_ne_m2": None if within is None else within.tolist(),
           "between_flight_covariance_ne_m2": None if between is None else between.tolist(),
           "calibration_height_m": float(np.mean(heights[:nf])) if heights else None,
           "between_flight_tilt_sigma_deg": (math.degrees(math.sqrt(np.trace(between) / 2)
                                                          / float(np.mean(heights[:nf])))
                                             if heights and between is not None else None),
           "fit_flights": list(cal.fit_flights), "holdout_flights": list(cal.holdout_flights),
           "holdout_nees_coverage_95pct": cal.holdout_nees_coverage_95pct,
           "holdout_mean_nees_2d": cal.holdout_mean_nees_2d,
           "holdout_false_fix_rate": cal.holdout_false_fix_rate_among_accepted,
           "fit_false_fix_rate": float(np.mean([f.false_fixes / max(1, f.false_fixes + len(f.errors_ne_m))
                                                 for f in flights[:nf]])),
           "outlier_rule": rule,
           "places": [f.get("places") for f in raw],
           "map_heading": heading_calibration(args.flights, nf),
           "place_model": place_model,
           "quality_model": quality_calibration(args.flights, cal.bias_ne_m, nf, raw),
           "excluded_test_zone_fixes": excluded,
           "test_zone": TEST_ZONE if excluded is not None else None,
           # The companion uses quality_model only if this is true.
           "use_quality_model": bool(args.quality_model),
           "error_field": field_out,
           "use_error_field": field_out is not None,
           "per_source_bias_ne_m": per_source["bias"] if per_source else None,
           "per_source_covariance_ne_m2": per_source["cov"] if per_source else None,
           "per_source_check": per_source["check"] if per_source else None,
           "holdout_details": cal.holdout_flight_details,
           "independently_validated_for_flight": False,
           "note": "simulated flights (Gazebo, idealised camera)"}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "map_calibration.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("bias_ne_m", "sigma_ne_m", "covariance_model", "holdout_check")}))


def read_map_log(path: Path) -> list[dict]:
    rows = []
    lines = path.read_text().splitlines()
    head = lines[0].split(",")
    for ln in lines[1:]:
        rows.append(dict(zip(head, ln.split(","))))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("px4_dir", type=Path)
    ap.add_argument("--mode", choices=("calibrate", "combine", "test", "heading"), required=True)
    ap.add_argument("--frame-time-shift-ms", type=float, default=0.0,
                    help="companion: camera frame stamp shift (tools/measure_frame_time_shift.py)")
    ap.add_argument("--attitude-stamp", choices=("publish", "sample"), default="sample",
                    help="companion: VO attitude from ATTITUDE_QUATERNION (publish) or ODOMETRY (sample)")
    ap.add_argument("--vo-surface-depth", action="store_true",
                    help="companion: VO feature depths on the DSM (with --dsm-surface)")
    ap.add_argument("--vo-window-pairs", type=int, default=1,
                    help="companion: mean VO velocity over this many frame pairs")
    ap.add_argument("--no-ev-pos-repeat-scale", action="store_true",
                    help="companion: position variance not scaled for its repetition (see gz_vo_companion.py)")
    ap.add_argument("--map-weak-match", type=float, nargs=2, metavar=("MARGIN", "PEAK"),
                    help="companion: reject NCC matches below both (see gz_vo_companion.py)")
    ap.add_argument("--map-hold", type=float, nargs=2, metavar=("AGE_S", "MOVED_M"),
                    help="companion: hold parameters (default in gz_vo_companion.py: 30 s, 3 m)")
    ap.add_argument("--no-map-hold", action="store_true",
                    help="companion: no map position hold without fixes (see gz_vo_companion.py)")
    ap.add_argument("--map-patch-at-boresight", action="store_true",
                    help="companion: map patch centred on the optical axis (see gz_vo_companion.py)")
    ap.add_argument("--map-max-tilt-rad", type=float, default=0.12, help="companion: map matching tilt limit")
    ap.add_argument("--vo-gap-bridge", type=Path,
                    help="companion: VO-gap bridge calibration (JSON; default: the companion's)")
    ap.add_argument("--no-vo-gap-bridge", action="store_true", help="companion: no VO-gap bridge")
    ap.add_argument("--vo-outages-s", type=float, nargs="+", metavar="LEN",
                    help="calibration: injected VO dropouts on the test route, these lengths in turn "
                         "(sim s), --vo-outage-off-s apart (gap bridge calibration flights, not tests)")
    ap.add_argument("--vo-outage-off-s", type=float, default=6.0)
    ap.add_argument("--vo-max-tilt-rad", type=float, default=0.25, help="companion: VO frame-pair tilt limit")
    ap.add_argument("--wind-ne-m-s", type=float, nargs=2, metavar=("NORTH", "EAST"),
                    help="steady wind from reaching altitude on (drag force on the model, tools/gz_wind.py)")
    ap.add_argument("--wind-gust-sigma", type=float, default=0.0, help="gust speed sigma per axis, m/s")
    ap.add_argument("--wind-seed", type=int, default=0)
    ap.add_argument("--map-outage-legs", type=int, nargs=2, metavar=("FIRST", "LAST"),
                    help="test: no map matching from route leg FIRST to the end of leg LAST (1-based); "
                         "diagnostic MAP_OUTAGE (check 3 then includes the outage)")
    ap.add_argument("--save-every", type=int, default=75,
                    help="companion: save every N-th camera frame (1: all, for offline VO replay)")
    ap.add_argument("--vo-calibration", type=Path,
                    help="companion: camera-VO variance scale (tools/calibrate_vo_covariance.py)")
    ap.add_argument("--ev-yaw", action="store_true",
                    help="send the map-referenced heading to EKF2 (EKF2_EV_CTRL bit 3) in any mode")
    ap.add_argument("--mag-decl-error-deg", type=float, default=12.0,
                    help="heading: declination error from boot (EKF2_DECL_TYPE=0), e.g. a wrong "
                         "declination or a magnetic disturbance at the take-off site")
    ap.add_argument("--wmm-declination-deg", type=float, default=3.41,
                    help="heading: declination PX4 derives from WMM at the Gazebo world origin")
    ap.add_argument("--flights", type=Path, nargs="*", default=(),
                    help="combine: calibration flight directories, --fit-flights fit then the rest held out")
    ap.add_argument("--test-route", choices=sorted(TEST_ROUTES), default="box",
                    help="test: legs of the absolute-error route (all inside TEST_ZONE)")
    ap.add_argument("--exclude-test-zone", action="store_true",
                    help=f"combine: leave out calibration fixes in the test zone ({TEST_ZONE})")
    ap.add_argument("--error-field", action="store_true",
                    help="combine: kriged map-offset field (bias and variance at each fix position)")
    ap.add_argument("--field-drift", choices=("constant", "linear"), default="constant",
                    help="combine --error-field: mean of the field, constant (ordinary kriging) or "
                         "linear in N/E (universal kriging)")
    ap.add_argument("--flight-quantile", type=float,
                    help="combine: scale R so that this quantile of the cross-validated held-out "
                         "flights' mean NEES is 2 (flight-level conservative covariance)")
    ap.add_argument("--fit-flights", type=int, default=3,
                    help="combine: the first N flights are fitted, the rest held out")
    ap.add_argument("--quality-model", action="store_true",
                    help="combine: the companion uses the match-quality covariance model "
                         "(otherwise it is a diagnostic only)")
    ap.add_argument("--place-weighted", action="store_true",
                    help="combine: bias and covariance from the place-weighted model (one weight per "
                         "20 m place; fixes in a place are not independent)")
    ap.add_argument("--isotropic", action="store_true",
                    help="combine: sigma^2 = trace(C)/2 in both axes (shape of C not identifiable "
                         "from fewer than 5 fit flights); held-out NEES recomputed with it")
    ap.add_argument("--out", type=Path, default=Path("build/gz_map_companion"))
    ap.add_argument("--calibration", type=Path, help="map_calibration.json from --mode calibrate")
    ap.add_argument("--focal-error", type=float,
                    help="injected focal-length error (VO scale fault); default 0.05 textured, 0 swiss "
                         "(the long swiss route shows the natural VO drift, and NCC needs the true scale)")
    ap.add_argument("--georef-error", type=float, nargs=2, default=(0.15, -0.10))
    ap.add_argument("--companion-python", default=sys.executable)
    ap.add_argument("--dsm-surface", action="store_true",
                    help="swiss: project frames onto orthophoto_surface.npy in --swiss-dir (DSM) in the localizer")
    ap.add_argument("--world-3d", action="store_true",
                    help="swiss: fly in ofnav_swiss3d (DSM mesh, tools/swiss_scene_3d.py) instead of the flat plane")
    ap.add_argument("--scene", choices=("textured", "swiss"), default="textured",
                    help="swiss: real SWISSIMAGE ground and onboard map (tools/swiss_scene.py)")
    ap.add_argument("--swiss-dir", type=Path, help="output of tools/swiss_scene.py (orthophoto + georef)")
    ap.add_argument("--map-image-fallback",
                    help="swiss: second map in --swiss-dir tried when the first gives no fix (per-source calibration)")
    ap.add_argument("--map-image", default="orthophoto.png",
                    help="swiss: map image in --swiss-dir (e.g. orthophoto_ground.png from swiss_ground_map.py)")
    ap.add_argument("--altitude", type=float, help="take-off altitude, m (default 5 textured, 40 swiss)")
    ap.add_argument("--calib-walk-seed", type=int,
                    help="calibrate: random walk over the map with this seed (legs avoid the test-route quadrant)")
    ap.add_argument("--calib-walk-legs", type=int, default=5)
    ap.add_argument("--calib-walk-sector-deg", type=float, nargs=2, default=(90.0, 360.0),
                    help="calibrate walk: leg directions (0 forward = east, 90 right = south). "
                         "180 270 (west to north) keeps every leg out of TEST_ZONE. 180 360 does not: "
                         "legs near 360 head east, into the zone (series P flights 5-6); 90 360: "
                         "series M/N, legs near 90 deg ran along the zone's south strip")
    ap.add_argument("--calib-walk-in-test-zone", action="store_true",
                    help="calibrate walk: over the test routes instead of away from them (map calibration "
                         "of the test zone itself; a leg ends on leaving CALIB_ZONE_BOX). Its result "
                         "is a zone-calibrated setup, reported apart from the zone-independent one")
    ap.add_argument("--calib-walk-bound-m", type=float, default=170.0,
                    help="calibrate walk: a leg ends early beyond this distance from the scene centre")
    ap.add_argument("--calib-walk-leg-s", type=float, default=12.0)
    ap.add_argument("--calib-direction-deg", type=float, default=0.0,
                    help="calibrate: direction of the two legs (0: forward then right). Each calibration "
                         "flight of a series uses its own (e.g. 72 deg apart) so the map error is "
                         "sampled at different places, not the same three each time")
    ap.add_argument("--camera-hz", type=int, default=15)
    args = ap.parse_args()
    if args.mode in ("test", "heading") and not args.calibration:
        ap.error("--mode test/heading needs --calibration")
    if args.map_outage_legs:
        first, last = args.map_outage_legs
        if args.mode not in ("test", "heading") or not 1 <= first <= last <= len(TEST_ROUTES[args.test_route]):
            ap.error(f"--map-outage-legs needs a test route and 1 <= FIRST <= LAST <= "
                     f"{len(TEST_ROUTES[args.test_route])}")
    if not 0.0 < args.vo_max_tilt_rad <= 0.4:
        ap.error("--vo-max-tilt-rad must be in (0, 0.4]")
    if args.calib_walk_in_test_zone and (args.mode != "calibrate" or args.calib_walk_seed is None):
        ap.error("--calib-walk-in-test-zone needs --mode calibrate and --calib-walk-seed")
    if args.vo_outages_s and (args.mode not in ("test", "heading") or args.map_outage_legs
                              or not all(0 < x <= 10 for x in args.vo_outages_s) or args.vo_outage_off_s <= 0):
        ap.error("--vo-outages-s needs a test route without --map-outage-legs, lengths in (0, 10] s")
    params = dict(PARAMS, EKF2_EV_CTRL="13") if args.ev_yaw else dict(PARAMS)
    if args.mode == "heading":
        # A 3D magnetometer with estimated earth field absorbs a declination
        # change made in flight within minutes (checked): the error must be
        # present at alignment, as with a wrong declination at the take-off site.
        params.update(EKF2_DECL_TYPE="0",
                      EKF2_MAG_DECL=f"{args.wmm_declination_deg + args.mag_decl_error_deg:.3f}")
    if args.mode == "combine":
        combine(args)
        return
    if args.scene == "swiss" and not args.swiss_dir:
        ap.error("--scene swiss needs --swiss-dir")
    altitude = args.altitude or (40.0 if args.scene == "swiss" else TAKEOFF_M)
    if args.focal_error is None:
        args.focal_error = 0.0 if args.scene == "swiss" else 0.05
    params["MIS_TAKEOFF_ALT"] = f"{altitude:.1f}"
    world = ("ofnav_swiss3d" if args.world_3d else "ofnav_swiss") if args.scene == "swiss" else gz.TEXTURED_WORLD
    # Route scale: legs long enough to cross several camera footprints.
    stick, leg_s, sample_s = (400, 15, 3) if args.scene == "swiss" else (300, 4, 2)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))
    args.out.mkdir(parents=True, exist_ok=True)
    px4_dir = args.px4_dir.resolve()
    vo_flight.install_vehicle(px4_dir, rate_hz=args.camera_hz)
    if args.scene == "swiss":
        # Ground: SWISSIMAGE 2025; map: SWISSIMAGE 2019, exact LV95 georef.
        map_png, georef = args.swiss_dir / args.map_image, args.swiss_dir / "orthophoto_georef.json"
        args.georef_error = (0.0, 0.0)
    else:
        gz.install_textured_world(px4_dir)
        map_png, georef = make_orthophoto(px4_dir, args.out, tuple(args.georef_error))
    control = args.out / "map_control.json"
    control.write_text(json.dumps({"shift_ne_m": [0.0, 0.0]}))
    report: dict = {"type": f"PX4_SITL_GAZEBO_COMPANION_MAP_{args.mode.upper()}", "params": params,
                    "ev_yaw": bool(args.ev_yaw),
                    "focal_error": args.focal_error, "georef_error_ne_m": list(args.georef_error),
                    "checks": {}, "segments": {}}
    started_wall = time.time()
    px4 = gz.Px4(px4_dir, args.out / "px4_gz.log", world, params, model=f"gz_{vo_flight.MODEL}")
    link = gz.Link()
    gz.SIM_CLOCK = link
    clog_path = args.out / "companion.log"
    clog = clog_path.open("w")
    companion = None
    wind = None
    ok_all = True

    def check(name: str, cond: bool, **info) -> None:
        nonlocal ok_all
        ok_all = ok_all and bool(cond)
        report["checks"][name] = {"pass": bool(cond), **info}
        print(f"{'PASS' if cond else 'FAIL'} {name} {info}", flush=True)

    def truth_est():
        return np.array(xy(topic(px4, "vehicle_local_position_groundtruth"))), \
            np.array(xy(topic(px4, "vehicle_local_position")))

    def move(pitch: int, roll: int, seconds: float) -> None:
        link.pitch, link.roll = pitch, roll
        gz.sim_sleep(seconds)
        link.pitch = link.roll = 0

    try:
        if not gz.wait(lambda: link.m.target_system != 0, 180):
            raise RuntimeError("no MAVLink heartbeat from PX4")
        cmd = [args.companion_python, str(Path(__file__).with_name("gz_vo_companion.py")),
               "--world", world, "--camera-topic", vo_flight.CAMERA_TOPIC,
               "--log", str(args.out / "companion_vo.csv"), "--focal-error", str(args.focal_error),
               "--orthophoto", str(map_png), "--map-georef", str(georef),
               "--map-log", str(args.out / "companion_map.csv"), "--map-control", str(control),
               "--save-frames", str(args.out / "frames"), "--save-every", str(args.save_every)]
        if args.vo_calibration:
            cmd += ["--vo-calibration", str(args.vo_calibration.resolve())]
        cmd += ["--attitude-stamp", args.attitude_stamp]
        if args.map_max_tilt_rad != 0.12:
            cmd += ["--map-max-tilt-rad", str(args.map_max_tilt_rad)]
        if args.vo_gap_bridge:
            cmd += ["--vo-gap-bridge", str(args.vo_gap_bridge.resolve())]
        if args.no_vo_gap_bridge:
            cmd += ["--no-vo-gap-bridge"]
        if args.vo_max_tilt_rad != 0.25:
            cmd += ["--vo-max-tilt-rad", str(args.vo_max_tilt_rad)]
        if args.map_patch_at_boresight:
            cmd += ["--map-patch-at-boresight"]
        if args.no_ev_pos_repeat_scale:
            cmd += ["--no-ev-pos-repeat-scale"]
        if args.map_weak_match:
            cmd += ["--map-weak-match", *map(str, args.map_weak_match)]
        if args.map_hold:
            cmd += ["--map-hold", *map(str, args.map_hold)]
        if args.no_map_hold:
            cmd += ["--no-map-hold"]
        if args.vo_window_pairs != 1:
            cmd += ["--vo-window-pairs", str(args.vo_window_pairs)]
        if args.vo_surface_depth:
            if not args.dsm_surface:
                raise SystemExit("--vo-surface-depth needs --dsm-surface")
            cmd += ["--vo-surface-depth"]
        if args.frame_time_shift_ms:
            cmd += ["--frame-time-shift-ms", str(args.frame_time_shift_ms)]
        cmd += ["--map-mode", "observe"] if args.mode == "calibrate" else \
            ["--map-mode", "fuse", "--map-calibration", str(args.calibration.resolve())]
        if args.ev_yaw:
            cmd += ["--map-yaw"]
        if args.scene == "swiss":
            cmd += ["--map-method", "ncc"]
            if args.dsm_surface:
                cmd += ["--surface", str(args.swiss_dir / "orthophoto_surface.npy")]
            if args.map_image_fallback:
                cmd += ["--orthophoto-fallback", str(args.swiss_dir / args.map_image_fallback)]
        # The companion starts after the preflight gate: its start-up load
        # (orthophoto features) during EKF2 initialisation coincided with
        # failed magnetometer alignment in SITL.
        # The heading mode starts with a deliberate heading error: the SITL
        # sanity threshold is widened to keep that start (still rejecting the
        # 30-50 deg SITL artefacts).
        report["preflight"] = vo_flight.wait_estimator_settled(
            px4, max_heading_error_deg=args.mag_decl_error_deg + 8.0 if args.mode == "heading" else 10.0)
        companion = subprocess.Popen(cmd, stdout=clog, stderr=subprocess.STDOUT)
        gz.sim_sleep(5)

        def armed():
            link.command(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 1)
            return gz.wait(lambda: link.state["armed"], 3)
        if not gz.wait(armed, 180, 1):
            raise RuntimeError("arming failed: " + px4.cmd("commander", "check"))
        z_home = link.state["z"]
        link.command(mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, float("nan"),
                     float("nan"), float("nan"), float("nan"))
        if not gz.wait(lambda: link.state["z"] - z_home > altitude - 0.3, 120):
            raise RuntimeError(f"takeoff failed dz={link.state['z'] - z_home}")
        report["takeoff_us"] = int(link.boot_s * 1e6)
        if args.wind_ne_m_s:
            import gz_wind
            wind = gz_wind.WindDisturbance(world, f"{vo_flight.MODEL}_0", args.wind_ne_m_s,
                                           args.wind_gust_sigma, seed=args.wind_seed,
                                           log_path=args.out / "wind.csv", clock=lambda: link.boot_s)
            wind.start()
            report["wind"] = {"ne_m_s": list(args.wind_ne_m_s), "gust_sigma_m_s": args.wind_gust_sigma,
                              "seed": args.wind_seed,
                              "force_ne_n": gz_wind.drag_force_ne(args.wind_ne_m_s).round(3).tolist(),
                              "start_us": int(link.boot_s * 1e6)}
        gz.sim_sleep(4)
        link.set_mode(3)
        gz.wait(lambda: link.mode() == "POSCTL", 10)
        gz.sim_sleep(5)

        if args.mode == "calibrate":
            # One flight: three places over the map, slow motions in each.
            if args.calib_walk_seed is not None:
                # Random walk over the map: the map error varies from place
                # to place (series E/F: fix error on the test route twice the
                # calibrated one), so the places must be a sample of the
                # map, not of the start area. Directions avoid the quadrant
                # of the test route (forward/right = east/south at the
                # take-off heading): a in (90, 360) deg.
                # Successive directions differ by at most +-45 deg, so the
                # legs carry the vehicle away from the start instead of
                # cancelling (independent directions: 11-30 m span, series G).
                rng = np.random.default_rng(args.calib_walk_seed)
                lo, hi = args.calib_walk_sector_deg
                angles = [rng.uniform(lo, hi)]
                for _ in range(args.calib_walk_legs - 1):
                    angles.append(float(np.clip(angles[-1] + rng.uniform(-45.0, 45.0), lo, hi)))
                angles = np.asarray(angles)
                legs = [(0, 0)] + [(round(stick * math.cos(math.radians(x))), round(stick * math.sin(math.radians(x))))
                                   for x in angles]
                leg_s = args.calib_walk_leg_s
                report["calib_walk"] = {"seed": args.calib_walk_seed, "directions_deg": angles.round(1).tolist(),
                                        "leg_s": leg_s}
            else:
                a = math.radians(args.calib_direction_deg)
                legs = ((0, 0), (round(stick * math.cos(a)), round(stick * math.sin(a))),
                        (round(-stick * math.sin(a)), round(stick * math.cos(a))))
                leg_s = 4 if args.scene == "textured" else 8
            report["calib_direction_deg"] = args.calib_direction_deg
            for pitch, roll in legs:
                if pitch or roll:
                    if args.calib_walk_seed is not None:
                        # Stay over the scene: a leg ends early beyond
                        # calib_walk_bound_m from the centre (GNSS-aided
                        # position; series M flight 5 left the 400 m scene).
                        target = None
                        if args.calib_walk_in_test_zone:
                            # Towards a random point of the box (forward = east, right = south).
                            (n_lo, n_hi), (e_lo, e_hi) = CALIB_ZONE_BOX
                            target = (rng.uniform(n_lo + 5, n_hi - 5), rng.uniform(e_lo + 5, e_hi - 5))
                            a = math.atan2(-(target[0] - link.state["x"]), target[1] - link.state["y"])
                            pitch, roll = round(stick * math.cos(a)), round(stick * math.sin(a))
                            report.setdefault("calib_walk_zone_targets", []).append([round(target[0], 1),
                                                                                     round(target[1], 1)])
                        link.pitch, link.roll = pitch, roll
                        t_leg = link.boot_s
                        while link.boot_s - t_leg < leg_s:
                            gz.sim_sleep(1)
                            if math.hypot(link.state["x"], link.state["y"]) > args.calib_walk_bound_m:
                                report.setdefault("calib_walk_bounded_legs", 0)
                                report["calib_walk_bounded_legs"] += 1
                                break
                            if target is not None:
                                n_, e_ = link.state["x"], link.state["y"]
                                if (math.hypot(n_ - target[0], e_ - target[1]) < 4.0
                                        or not (n_lo <= n_ <= n_hi and e_lo <= e_ <= e_hi)):
                                    break
                                continue
                            # Nor over the test routes (their fixes are not
                            # used in combine anyway): beyond the start area.
                            if (in_test_zone(link.state["x"], link.state["y"])
                                    and math.hypot(link.state["x"], link.state["y"]) > 25.0):
                                report.setdefault("calib_walk_test_zone_legs", 0)
                                report["calib_walk_test_zone_legs"] += 1
                                break
                        link.pitch = link.roll = 0
                    else:
                        move(pitch, roll, leg_s)
                    gz.sim_sleep(4)
                for p, r in ((120, 0), (0, 120), (-120, 0), (0, -120)):
                    move(p, r, 2)
                    gz.sim_sleep(2)
            report["segments"]["flight"] = [report["takeoff_us"] + 8_000_000, int(link.boot_s * 1e6)]
        else:
            # 1: map lock while GNSS holds. The same small motions as in the
            # calibration flights: hovering over one spot repeats one view,
            # and over the swiss start point that view stayed ambiguous
            # (a second NCC peak 5.5 m away at another heading, margin
            # 0.04-0.09 against 0.05) for the whole check.
            for p, r in ((120, 0), (0, 120), (-120, 0), (0, -120)):
                move(p, r, 2)
                gz.sim_sleep(2)
            locked = gz.wait(lambda: latest_status(clog_path).get("map_anchor", {}).get("locked"), 60, 1)
            gz.sim_sleep(3)
            f1 = vo_flight.flags_and_validity(px4)
            f1["cs_ev_pos"] = topic(px4, "estimator_status_flags").get("cs_ev_pos")
            check("1_map_lock_with_gnss", bool(locked) and f1["cs_ev_pos"] == "True" and f1["cs_gnss_pos"] == "True",
                  **f1, anchor=latest_status(clog_path).get("map_anchor"))
            # 2: GNSS lost.
            px4.cmd("param", "set", "EKF2_GPS_CTRL", "0")
            report["gnss_off_us"] = int(link.boot_s * 1e6)
            gz.sim_sleep(5)
            f2 = vo_flight.flags_and_validity(px4)
            f2["cs_ev_pos"] = topic(px4, "estimator_status_flags").get("cs_ev_pos")
            check("2_gnss_lost_ekf2_on_map_and_vo", f2["cs_ev_pos"] == "True" and f2["cs_ev_vel"] == "True"
                  and f2["cs_gnss_pos"] == "False" and f2["xy_valid"] == "True", **f2)
            if args.mode == "heading":
                heading_phase(args, px4, link, report, check)
            # 3: box pattern, absolute error.
            errors, legs = [], []
            report["test_route"] = args.test_route
            report["route_start_us"] = int(link.boot_s * 1e6)
            outage = args.map_outage_legs
            vo_out = {"on": False, "next_s": link.boot_s + args.vo_outage_off_s, "k": 0}
            report["vo_outages_us"] = []

            def vo_outage_tick() -> None:
                # Injected VO dropouts (calibration of the gap bridge).
                if not args.vo_outages_s or link.boot_s < vo_out["next_s"]:
                    return
                if vo_out["on"]:
                    vo_out.update(on=False, next_s=link.boot_s + args.vo_outage_off_s)
                    report["vo_outages_us"][-1].append(int(link.boot_s * 1e6))
                else:
                    length = args.vo_outages_s[vo_out["k"] % len(args.vo_outages_s)]
                    vo_out.update(on=True, next_s=link.boot_s + length, k=vo_out["k"] + 1)
                    report["vo_outages_us"].append([int(link.boot_s * 1e6)])
                control.write_text(json.dumps({"shift_ne_m": [0.0, 0.0], "vo_outage": vo_out["on"]}))

            def route_sleep(s: float) -> None:
                # Fine steps while VO dropouts are injected (lengths below the 2-3 s samples).
                if not args.vo_outages_s:
                    gz.sim_sleep(s)
                    return
                t_end = link.boot_s + s
                while link.boot_s < t_end - 1e-3:
                    gz.sim_sleep(min(0.25, t_end - link.boot_s))
                    vo_outage_tick()
            for leg_no, (fp, fr) in enumerate(TEST_ROUTES[args.test_route], 1):
                if outage and leg_no == outage[0]:
                    control.write_text(json.dumps({"shift_ne_m": [0.0, 0.0], "outage": True}))
                    report["map_outage_start_us"] = int(link.boot_s * 1e6)
                pitch, roll = round(fp * stick), round(fr * stick)
                link.pitch, link.roll = pitch, roll
                t_leg = link.boot_s
                while link.boot_s - t_leg < leg_s:
                    route_sleep(sample_s)
                    truth, est = truth_est()
                    errors.append(float(np.linalg.norm(est - truth)))
                link.pitch = link.roll = 0
                for _ in range(4):
                    route_sleep(2)
                    truth, est = truth_est()
                    errors.append(float(np.linalg.norm(est - truth)))
                legs.append({"truth_ne_m": truth.round(2).tolist(), "estimate_ne_m": est.round(2).tolist()})
                if outage and leg_no == outage[1]:
                    control.write_text(json.dumps({"shift_ne_m": [0.0, 0.0]}))
                    report["map_outage_end_us"] = int(link.boot_s * 1e6)
            if vo_out["on"]:
                control.write_text(json.dumps({"shift_ne_m": [0.0, 0.0]}))
                report["vo_outages_us"][-1].append(int(link.boot_s * 1e6))
            report["route_legs"] = legs
            report["route_end_us"] = int(link.boot_s * 1e6)
            # Limit fixed before the flight: 3 sigma of the calibrated map error, at least 0.5 m.
            cal = json.loads(args.calibration.read_text())
            limit = max(0.5, 3.0 * math.sqrt(max(cal["covariance_ne_m2"][0][0], cal["covariance_ne_m2"][1][1])))
            report["route_error_limit_m"] = round(limit, 3)
            check("3_route_absolute_error_on_map", max(errors) < limit, max_abs_error_m=round(max(errors), 3),
                  mean_abs_error_m=round(float(np.mean(errors)), 3), samples=len(errors))
        if args.mode == "test":
            # 4: map fault.
            def truth_ne():
                return truth_est()[0]
            def means(seconds=DRIFT_MEAN_S, step_s=0.5):
                # Mean true position and mean estimate error over the same samples.
                pts = []
                for i in range(int(round(seconds / step_s)) + 1):
                    if i:
                        gz.sim_sleep(step_s)
                    truth, est = truth_est()
                    pts.append(np.concatenate([truth, est - truth]))
                m = np.mean(pts, axis=0)
                return m[:2], m[2:]
            settle_s = wait_settled(truth_ne, gz.sim_sleep)
            truth0, err0 = means()
            before = latest_status(clog_path).get("map_anchor", {}).get("counts", {})
            control.write_text(json.dumps({"shift_ne_m": [0.0, 3.0]}))
            report["fault_start_us"] = int(link.boot_s * 1e6)
            worst = 0.0
            for _ in range(9):
                gz.sim_sleep(2)
                truth, est = truth_est()
                worst = max(worst, float(np.linalg.norm(est - truth)))
            # Last 2 s of the 20-s fault: mean true position and estimate error.
            truth, err1 = means()
            t_est, est = truth_est()
            worst = max(worst, float(np.linalg.norm(est - t_est)))
            during = latest_status(clog_path).get("map_anchor", {}).get("counts", {})
            control.write_text(json.dumps({"shift_ne_m": [0.0, 0.0]}))
            report["fault_end_us"] = int(link.boot_s * 1e6)
            fused_during = during.get("FIX_FUSED", 0) - before.get("FIX_FUSED", 0)
            rejected = sum(during.get(key, 0) - before.get(key, 0)
                           for key in ("FIX_REJECTED_NIS", "FIX_REJECTED_STEP"))
            drift = float(np.linalg.norm(truth - truth0))
            # Integrity measure: how far the estimate moved away from the truth.
            # The true drift also holds the vehicle's own wander in gusts
            # (strong gusts: up to 1.3 m with the estimate moving 0.2 m).
            pull = float(np.linalg.norm(err1 - err0))
            gz.sim_sleep(10)
            after = latest_status(clog_path).get("map_anchor", {}).get("counts", {})
            refused = after.get("FIX_FUSED", 0) - during.get("FIX_FUSED", 0)
            # Fused/rejected counts are taken from the companion map log after
            # the flight (status lines are 5 s apart); vehicle behaviour here.
            report["fault_live"] = {"max_abs_error_during_fault_m": round(worst, 3),
                                    "true_drift_during_fault_m": round(drift, 3),
                                    "estimate_pull_during_fault_m": round(pull, 3),
                                    "settle_s": settle_s,
                                    "status_counts_delta": {"fused": fused_during, "rejected": rejected,
                                                            "fused_after": refused}}
        link.set_mode(4, 6)
        landing = gz.wait(lambda: link.mode() == "AUTO.LAND", 10)
        landed = gz.wait(lambda: not link.state["armed"], 120)
        check("5_land", landing and landed, disarmed=landed)
    except vo_flight.SitlStartRejected as exc:
        ok_all = False
        report["sitl_start_rejected"] = str(exc)
        print("SITL_START_REJECTED", exc, flush=True)
    except Exception as exc:  # report the failure, never hide it
        ok_all = False
        report["error"] = repr(exc)
        print("ERROR", exc, flush=True)
    finally:
        if wind is not None:
            wind.stop()
        link.close()
        if companion is not None:
            companion.terminate()
            try:
                companion.wait(timeout=10)
            except subprocess.TimeoutExpired:
                companion.kill()
        clog.close()
        px4.stop()
    report["companion_final_status"] = latest_status(clog_path)
    if "sitl_start_rejected" in report:
        (args.out / f"gz_map_companion_{args.mode}.json").write_text(json.dumps(report, indent=2))
        raise SystemExit(3)

    try:
        analyse(args, px4_dir, started_wall, report)
    except Exception as exc:  # analysis failure is a failure
        ok_all = False
        report["analysis_error"] = repr(exc)
        print("ANALYSIS ERROR", exc, flush=True)
    ok_all = ok_all and report.get("analysis_ok", True)
    report["passed"] = ok_all
    (args.out / f"gz_map_companion_{args.mode}.json").write_text(json.dumps(report, indent=2))
    raise SystemExit(0 if ok_all else 1)


def analyse(args, px4_dir: Path, started_wall: float, report: dict) -> None:
    d, ulog = ulog_truth(px4_dir, started_wall)
    truth, est = d["vehicle_local_position_groundtruth"], d["vehicle_local_position"]
    report["ulog"] = ulog
    if report.get("takeoff_us"):
        report["ekf2_heading_error"] = heading_error_deg(d, report["takeoff_us"], report.get("gnss_off_us"))
    rows = read_map_log(args.out / "companion_map.csv")
    offset = int(report["companion_final_status"].get("time_offset_us") or 0)
    acc = [r for r in rows if r["localizer_status"].endswith("ACCEPTED_UNVALIDATED")]
    report["map_fixes"] = {"attempted": len(rows), "accepted_by_localizer": len(acc),
                           "median_process_ms": float(np.median([float(r["process_ms"]) for r in rows]))
                           if rows else None}
    if not acc:
        raise RuntimeError("no accepted map fixes")
    t = np.array([int(r["t_us"]) + offset for r in acc])
    fixes = np.array([[float(r["north_m"]) - float(r["injected_shift_n"]),
                       float(r["east_m"]) - float(r["injected_shift_e"])] for r in acc])
    err = fixes - interp(truth, t)
    report["ekf2_tilt_error_deg"] = tilt_error_deg(d, report.get("takeoff_us", 0))
    hdg = map_heading_errors_deg(d, acc, t)
    ok_h = np.isfinite(hdg)
    report["map_heading_error_deg"] = {
        "fixes_with_heading": int(ok_h.sum()), "fixes": len(hdg),
        "mean": round(float(np.mean(hdg[ok_h])), 2) if ok_h.any() else None,
        "std": round(float(np.std(hdg[ok_h])), 2) if ok_h.any() else None}
    if args.mode == "calibrate":
        t0, t1 = report["segments"]["flight"]
        sel = (t >= t0) & (t <= t1)
        e_all = err[sel]
        # Explicit outlier rule (fixed before the swiss flights): more than
        # FALSE_FIX_M from the flight median is a false fix. The covariance is
        # estimated from the rest; the rate is reported, the flight is kept
        # (re-flying until no outlier would hide them).
        places = None
        src_all = np.array([r.get("map_source") or "primary" for r in acc])[sel]
        if report.get("calib_walk") and len(set(src_all)) > 1:
            # Two maps (primary ground-only, fallback full): the place rule per
            # source, since their offsets differ (relief displacement).
            pos = interp(truth, t)[sel]
            inl = np.zeros(len(e_all), bool); ff = np.zeros(len(e_all), bool); fp = np.zeros(len(e_all), bool)
            n_pl = n_fpl = 0
            for src in sorted(set(src_all)):
                k = src_all == src
                pr = place_outlier_rule(e_all[k], pos[k])
                inl[k], ff[k], fp[k] = pr["inlier"], pr["false_fix"], pr["false_place"]
                n_pl += pr["places"]; n_fpl += pr["false_places"]
            places = {"inlier": inl, "false_fix": ff, "false_place": fp, "places": n_pl, "false_places": n_fpl}
            bad = ~inl
        elif report.get("calib_walk"):
            # Flights over many places (random walk): the place rule
            # (map_error_calibration.place_outlier_rule, thresholds fixed
            # before series H was re-analysed): 20 m places, a place more
            # than 3 m off is a false place, a fix more than 1 m from its
            # place median is a false fix.
            places = place_outlier_rule(e_all, interp(truth, t)[sel])
            bad = ~places["inlier"]
        else:
            bad = np.linalg.norm(e_all - np.median(e_all, axis=0), axis=1) > FALSE_FIX_M
        sub = [tuple(float(r.get(k, "nan") or "nan") for k in ("sub_nn", "sub_ne", "sub_ee")) for r in acc]
        false = int(np.sum(bad)) if places is None else int(np.sum(places["false_fix"]))
        e = e_all[~bad]
        pos_inl = interp(truth, t)[sel][~bad]
        total = sum(1 for r in rows if t0 <= int(r["t_us"]) + offset <= t1)
        out = {"name": args.out.name, "errors_ne_m": e.round(4).tolist(), "total_frames": total,
               "false_fixes": false, "mean_error_ne_m": e.mean(axis=0).round(3).tolist(),
               "std_error_ne_m": e.std(axis=0).round(4).tolist(),
               "ekf2_tilt_error_deg": report["ekf2_tilt_error_deg"],
               "outlier_rule": "place" if places is not None else f"|error - flight median| > {FALSE_FIX_M} m",
               "places": None if places is None else {
                   "places": places["places"], "false_places": places["false_places"],
                   "false_place_fixes": int(np.sum(places["false_place"])),
                   "fixes_in_small_places": int(np.sum(~(places["inlier"] | places["false_fix"]
                                                          | places["false_place"])))},
               "median_height_m": float(np.median([float(r["height_m"]) for r, s_ in zip(acc, sel) if s_])),
               # Map heading error of the same fixes (NaN: no heading from the search).
               "heading_errors_deg": [round(float(x), 3) for x in hdg[sel]],
               "map_source": src_all[~bad].tolist(),
               # True positions of the same fixes: places for the place-weighted model.
               "positions_ne_m": pos_inl.round(2).tolist(),
               # Sub-patch covariance of the same (inlier) fixes, for the
               # match-quality covariance model (NaN rows: not available).
               "subpatch_cov_ne_m2": [[[sub[k][0], sub[k][1]], [sub[k][1], sub[k][2]]]
                                      for k in np.flatnonzero(sel)[~bad]]}
        (args.out / "map_errors.json").write_text(json.dumps(out, indent=1))
        report["flight_map_errors"] = {k: v for k, v in out.items() if k != "errors_ne_m"}
        report["analysis_ok"] = len(e) >= 30
        print("MAP_ERRORS", report["flight_map_errors"], flush=True)
        return
    t_f0, t_f1 = report.get("fault_start_us"), report.get("fault_end_us")
    if t_f0 and t_f1:
        def in_fault(r):
            return float(r["injected_shift_n"]) != 0.0 or float(r["injected_shift_e"]) != 0.0
        shifted = [r for r in rows if in_fault(r)]
        after = [r for r in rows if int(r["t_us"]) + offset > t_f1 and not in_fault(r)]
        fused_shifted = sum(r["anchor_result"] == "FIX_FUSED" for r in shifted)
        rejected = sum(r["anchor_result"] in ("FIX_REJECTED_NIS", "FIX_REJECTED_STEP") for r in shifted)
        fused_after = sum(r["anchor_result"] == "FIX_FUSED" for r in after)
        live = report.get("fault_live", {})
        # Integrity: the estimate is not pulled by the shifted map (change of
        # estimate - truth over the fault < 0.5 m; the true drift is reported,
        # in strong gusts it is mostly the vehicle's wander). The absolute error during the fault includes the
        # error the estimate already had: same limit as check 3 (it was a
        # fixed 0.5 m, set for the 20-m scene with sigma 0.16 m).
        abs_limit = report.get("route_error_limit_m", 0.5)
        ok = (fused_shifted == 0 and rejected >= 5 and fused_after > 0
              and live.get("max_abs_error_during_fault_m", 9) < abs_limit
              and live.get("estimate_pull_during_fault_m", 9) < 0.5)
        report["checks"]["4_shifted_map_rejected"] = {
            "pass": ok, "shifted_fixes": len(shifted), "fused_shifted": fused_shifted,
            "rejected_total": rejected,
            "rejected_nis": sum(r["anchor_result"] == "FIX_REJECTED_NIS" for r in shifted),
            "rejected_step": sum(r["anchor_result"] == "FIX_REJECTED_STEP" for r in shifted),
            "min_nis_shifted": round(min((float(r["nis"]) for r in shifted if r["nis"] != "nan"), default=math.nan), 1),
            "fused_after_fault": fused_after, **{k: v for k, v in live.items() if k != "status_counts_delta"}}
        report["analysis_ok"] = ok
        print("PASS" if ok else "FAIL", "4_shifted_map_rejected", report["checks"]["4_shifted_map_rejected"], flush=True)
    # Check 3 also over the whole route from the ULog: the live samples are
    # 3 s apart and missed a 5.7 m excursion of 2 s (series P repeat, box_rf).
    c3 = report.get("checks", {}).get("3_route_absolute_error_on_map")
    t_r0, t_r1 = report.get("route_start_us"), report.get("route_end_us")
    if c3 is not None and t_r0 and t_r1:
        tg = truth["timestamp"]
        sel = (tg >= t_r0) & (tg <= t_r1)
        e_cont = np.linalg.norm(interp(est, tg[sel]) - interp(truth, tg[sel]), axis=1)
        limit = report["route_error_limit_m"]
        c3["max_abs_error_continuous_m"] = round(float(e_cont.max()), 3)
        c3["continuous_samples"] = int(sel.sum())
        c3["pass"] = bool(c3["pass"] and e_cont.max() < limit)
        report["analysis_ok"] = report.get("analysis_ok", True) and c3["pass"]
        print("PASS" if c3["pass"] else "FAIL", "3_route_absolute_error_on_map (continuous)", c3, flush=True)
    # Map outage (diagnostic, no pass/fail): EKF2 error without map fixes
    # and after they return.
    t_o0, t_o1 = report.get("map_outage_start_us"), report.get("map_outage_end_us")
    if t_o0 and t_o1 and t_r1:
        tg = truth["timestamp"]
        def err_between(a, b):
            sel = (tg >= a) & (tg <= b)
            return np.linalg.norm(interp(est, tg[sel]) - interp(truth, tg[sel]), axis=1) if sel.any() else np.array([np.nan])
        during = err_between(t_o0, t_o1)
        after = err_between(t_o1 + 10e6, t_r1)
        report["map_outage"] = {"duration_s": round((t_o1 - t_o0) * 1e-6, 1),
                                "error_at_start_m": round(float(err_between(t_o0, t_o0 + 0.5e6)[0]), 3),
                                "max_error_during_m": round(float(np.nanmax(during)), 3),
                                "error_at_end_m": round(float(during[-1]), 3),
                                "max_error_10s_after_end_to_route_end_m": round(float(np.nanmax(after)), 3)}
        print("MAP_OUTAGE", report["map_outage"], flush=True)
    # Test: VO-only track from GNSS loss to the end of the box, same flight.
    vo = np.genfromtxt(args.out / "companion_vo.csv", delimiter=",", names=True)
    t_off, t_end = report.get("gnss_off_us"), report.get("route_end_us")
    if t_off and t_end:
        sel = (vo["t_us"] >= t_off) & (vo["t_us"] <= t_end)
        tv, vn, ve = vo["t_us"][sel], vo["vn"][sel], vo["ve"][sel]
        dt = np.diff(tv, prepend=tv[0]) * 1e-6
        track = np.column_stack([np.cumsum(vn * dt), np.cumsum(ve * dt)])
        true_track = interp(truth, tv) - interp(truth, np.array([tv[0]]))
        est_track = interp(est, tv) - interp(est, np.array([tv[0]]))
        report["vo_only_vs_map"] = {
            "vo_only_final_error_m": round(float(np.linalg.norm(track[-1] - true_track[-1])), 3),
            "ekf2_map_final_error_m": round(float(np.linalg.norm(est_track[-1] - true_track[-1])), 3),
            "path_length_m": round(float(np.sum(np.linalg.norm(np.diff(true_track, axis=0), axis=1))), 2)}
        print("VO_ONLY_VS_MAP", report["vo_only_vs_map"], flush=True)
    # Fused fixes vs truth (bias-corrected by the anchor): residual statistics.
    fused = [r for r in acc if r["anchor_result"] == "FIX_FUSED"]
    report["fused_fix_count"] = len(fused)
    report["map_fix_error_ne_m"] = {"mean": err.mean(axis=0).round(3).tolist(),
                                   "std": err.std(axis=0).round(3).tolist()}
    # Diagnostic: NEES of the bias-corrected fixes with the calibrated
    # covariance (constant) and, if calibrated, the match-quality model.
    if args.calibration:
        cal = json.loads(args.calibration.read_text())
        e = err - np.asarray(cal["bias_ne_m"])
        c0 = np.asarray(cal["covariance_ne_m2"])
        n0 = np.einsum("ni,ij,nj->n", e, np.linalg.inv(c0), e)
        diag = {"constant_mean_nees": round(float(np.mean(n0)), 2),
                "constant_coverage_95": round(float(np.mean(n0 <= 5.991)), 3)}
        q = cal.get("quality_model")
        if q:
            nq = []
            for r, ei in zip(acc, e):
                c = [float(r.get(k, "nan") or "nan") for k in ("sub_nn", "sub_ne", "sub_ee")]
                if all(map(math.isfinite, c)):
                    rq = quality_covariance(q["floor_m2"], q["k2"], [[c[0], c[1]], [c[1], c[2]]])
                    nq.append(float(ei @ np.linalg.solve(rq, ei)))
            if nq:
                nq = np.asarray(nq)
                diag.update({"quality_mean_nees": round(float(np.mean(nq)), 2),
                             "quality_coverage_95": round(float(np.mean(nq <= 5.991)), 3), "quality_fixes": len(nq)})
        if cal.get("use_error_field") and cal.get("error_field"):
            from src.map_error_field import MapErrorField
            fld = MapErrorField.from_json(cal["error_field"])
            pos_t = interp(truth, t)
            # The map log holds the fixes after the companion removed the field
            # offset: err is already the residual against the field.
            nf_ = np.sum(err ** 2, axis=1) / fld.fix_variance(pos_t)
            diag.update({"field_mean_nees": round(float(np.mean(nf_)), 2),
                         "field_coverage_95": round(float(np.mean(nf_ <= 5.991)), 3),
                         "field_sigma_median_m": round(float(np.sqrt(np.median(fld.fix_variance(pos_t)))), 3)})
        # Which R the companion fused with (fix_covariance in gz_vo_companion).
        diag["covariance_used_by_companion"] = (
            "error_field" if cal.get("use_error_field") and cal.get("error_field")
            else "quality_model" if q and cal.get("use_quality_model") is True else "covariance_ne_m2")
        report["map_fix_consistency"] = diag
        print("MAP_FIX_CONSISTENCY", diag, flush=True)


if __name__ == "__main__":
    main()
