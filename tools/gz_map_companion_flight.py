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


def heading_error_deg(d: dict, t0_us: int) -> dict:
    """EKF2 heading minus ground truth, airborne part (diagnostic only)."""
    a, g = d["vehicle_attitude"], d["vehicle_attitude_groundtruth"]
    sel = a["timestamp"] >= t0_us
    truth = np.interp(a["timestamp"][sel], g["timestamp"], np.unwrap(np.radians(yaw_deg(g))))
    err = (np.radians(yaw_deg(a))[sel] - truth + np.pi) % (2 * np.pi) - np.pi
    return {"max_abs_deg": round(float(np.degrees(np.max(np.abs(err)))), 2),
            "mean_deg": round(float(np.degrees(np.mean(err))), 2)}


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
TEST_ROUTES = {
    # Legs of the test box as (pitch, roll) stick fractions; all stay in TEST_ZONE.
    "box": ((1, 0), (0, 1), (1, 0), (0, 1)),
    "box_rf": ((0, 1), (1, 0), (0, 1), (1, 0)),
    "diag": ((0.7, 0.7), (1, 0), (0.7, 0.7), (0, 1)),
}


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
               "--save-frames", str(args.out / "frames")]
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
                angles = [rng.uniform(90.0, 360.0)]
                for _ in range(args.calib_walk_legs - 1):
                    angles.append(float(np.clip(angles[-1] + rng.uniform(-45.0, 45.0), 90.0, 360.0)))
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
                        link.pitch, link.roll = pitch, roll
                        t_leg = link.boot_s
                        while link.boot_s - t_leg < leg_s:
                            gz.sim_sleep(1)
                            if math.hypot(link.state["x"], link.state["y"]) > args.calib_walk_bound_m:
                                report.setdefault("calib_walk_bounded_legs", 0)
                                report["calib_walk_bounded_legs"] += 1
                                break
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
            for fp, fr in TEST_ROUTES[args.test_route]:
                pitch, roll = round(fp * stick), round(fr * stick)
                link.pitch, link.roll = pitch, roll
                t_leg = link.boot_s
                while link.boot_s - t_leg < leg_s:
                    gz.sim_sleep(sample_s)
                    truth, est = truth_est()
                    errors.append(float(np.linalg.norm(est - truth)))
                link.pitch = link.roll = 0
                for _ in range(4):
                    gz.sim_sleep(2)
                    truth, est = truth_est()
                    errors.append(float(np.linalg.norm(est - truth)))
                legs.append({"truth_ne_m": truth.round(2).tolist(), "estimate_ne_m": est.round(2).tolist()})
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
            before = latest_status(clog_path).get("map_anchor", {}).get("counts", {})
            truth0, est0 = truth_est()
            control.write_text(json.dumps({"shift_ne_m": [0.0, 3.0]}))
            report["fault_start_us"] = int(link.boot_s * 1e6)
            worst = 0.0
            for _ in range(10):
                gz.sim_sleep(2)
                truth, est = truth_est()
                worst = max(worst, float(np.linalg.norm(est - truth)))
            during = latest_status(clog_path).get("map_anchor", {}).get("counts", {})
            control.write_text(json.dumps({"shift_ne_m": [0.0, 0.0]}))
            report["fault_end_us"] = int(link.boot_s * 1e6)
            fused_during = during.get("FIX_FUSED", 0) - before.get("FIX_FUSED", 0)
            rejected = sum(during.get(key, 0) - before.get(key, 0)
                           for key in ("FIX_REJECTED_NIS", "FIX_REJECTED_STEP"))
            drift = float(np.linalg.norm(truth - truth0))
            gz.sim_sleep(10)
            after = latest_status(clog_path).get("map_anchor", {}).get("counts", {})
            refused = after.get("FIX_FUSED", 0) - during.get("FIX_FUSED", 0)
            # Fused/rejected counts are taken from the companion map log after
            # the flight (status lines are 5 s apart); vehicle behaviour here.
            report["fault_live"] = {"max_abs_error_during_fault_m": round(worst, 3),
                                    "true_drift_during_fault_m": round(drift, 3),
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
        report["ekf2_heading_error"] = heading_error_deg(d, report["takeoff_us"])
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
        # Integrity: the vehicle is not pulled by the shifted map (true
        # drift < 0.5 m). The absolute error during the fault includes the
        # error the estimate already had: same limit as check 3 (it was a
        # fixed 0.5 m, set for the 20-m scene with sigma 0.16 m).
        abs_limit = report.get("route_error_limit_m", 0.5)
        ok = (fused_shifted == 0 and rejected >= 5 and fused_after > 0
              and live.get("max_abs_error_during_fault_m", 9) < abs_limit
              and live.get("true_drift_during_fault_m", 9) < 0.5)
        report["checks"]["4_shifted_map_rejected"] = {
            "pass": ok, "shifted_fixes": len(shifted), "fused_shifted": fused_shifted,
            "rejected_total": rejected,
            "rejected_nis": sum(r["anchor_result"] == "FIX_REJECTED_NIS" for r in shifted),
            "rejected_step": sum(r["anchor_result"] == "FIX_REJECTED_STEP" for r in shifted),
            "min_nis_shifted": round(min((float(r["nis"]) for r in shifted if r["nis"] != "nan"), default=math.nan), 1),
            "fused_after_fault": fused_after, **{k: v for k, v in live.items() if k != "status_counts_delta"}}
        report["analysis_ok"] = ok
        print("PASS" if ok else "FAIL", "4_shifted_map_rejected", report["checks"]["4_shifted_map_rejected"], flush=True)
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
        # Which R the companion fused with (fix_covariance in gz_vo_companion).
        diag["covariance_used_by_companion"] = "quality_model" if q and cal.get("use_quality_model") is True \
            else "covariance_ne_m2"
        report["map_fix_consistency"] = diag
        print("MAP_FIX_CONSISTENCY", diag, flush=True)


if __name__ == "__main__":
    main()
