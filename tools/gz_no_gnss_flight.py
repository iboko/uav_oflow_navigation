#!/usr/bin/env python3
"""Closed-loop flight WITHOUT GNSS on ofnav estimates: PX4 SITL + Gazebo.

gz_x500_flow has no GNSS (the airframe disables the simulated receiver).
The vehicle takes off on EKF2's built-in optical-flow fusion: below the
0.2 m envelope of the flow profile ofnav provides nothing, and an
unaided takeoff slides away. In flight the built-in fusion is switched
off (EKF2_OF_CTRL=0): from then on EKF2's ONLY horizontal aiding is
`ofnav start -e` (SITL builds only) — each accepted flow measurement as
body-FRD external-vision velocity (EKF2_EV_CTRL=4), the vertical body
component derived from the measured rangefinder height rate.

  1  after the switch: EKF2 fuses ofnav (cs_ev_vel), not its own flow
     (cs_opt_flow) and not GNSS (cs_gnss_pos), horizontal position valid
  2  POSCTL hover, sticks centred, 60 s of simulation: true horizontal
     drift (Gazebo ground truth) and EKF2 position error
  3  stick input forward inside the flow envelope (<= 1.5 m/s), release:
     the vehicle stops and holds; estimated vs true displacement. A second
     input beyond the envelope is recorded (ofnav rejects, EKF2 coasts)
  4  AUTO.LAND on the same estimates: landed and disarmed (the last
     0.2 m are below the ofnav envelope: EKF2 dead-reckons)

Ground truth comes from gz_bridge (vehicle_local_position_groundtruth),
never from the estimator. All waits are in simulation time. Writes a
JSON report; exit code != 0 on any failed check.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from pymavlink import mavutil

import gz_failure_scenarios as gz

PARAMS = (("EKF2_GPS_CTRL", "0"), ("EKF2_OF_CTRL", "1"), ("EKF2_EV_CTRL", "4"),
          ("EKF2_EV_DELAY", "0"), ("EKF2_HGT_REF", "0"), ("COM_ARM_WO_GPS", "1"),
          ("NAV_DLL_ACT", "0"), ("COM_RC_IN_MODE", "1"), ("NAV_RCL_ACT", "0"),
          ("COM_DISARM_LAND", "2"), ("MIS_TAKEOFF_ALT", "2.0"))
HOVER_S = 60.0
MAX_HOVER_DRIFT_M = 2.0       # true horizontal drift over the hover
MAX_ESTIMATE_ERROR_M = 2.0    # EKF2 displacement vs true displacement


def topic(px4: gz.Px4, name: str) -> dict:
    text = px4.cmd("listener", name, "-n", "1")
    out = {}
    for key, value in re.findall(r"^\s*([a-z_]+): (\S+)", text, re.M):
        try:
            out[key] = float(value)
        except ValueError:
            out[key] = value
    return out


def xy(d: dict) -> tuple[float, float]:
    return float(d.get("x", math.nan)), float(d.get("y", math.nan))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("px4_dir", type=Path)
    ap.add_argument("--out", type=Path, default=Path("build/gz_no_gnss"))
    ap.add_argument("--world", default=gz.TEXTURED_WORLD)
    ap.add_argument("--min-quality", default="60")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.world == gz.TEXTURED_WORLD:
        gz.install_textured_world(args.px4_dir.resolve())
    report: dict = {"type": "PX4_SITL_GAZEBO_FLIGHT_WITHOUT_GNSS_ON_OFNAV", "world": args.world,
                    "params": dict(PARAMS), "checks": {}, "samples": []}
    # Set at boot (PX4_PARAM_*): EKF2 starts without GNSS, no restart needed.
    px4 = gz.Px4(args.px4_dir.resolve(), args.out / "px4_gz.log", args.world, dict(PARAMS))
    link = gz.Link()
    gz.SIM_CLOCK = link
    ok_all = True

    def check(name: str, cond: bool, **info) -> None:
        nonlocal ok_all
        ok_all = ok_all and bool(cond)
        report["checks"][name] = {"pass": bool(cond), **info}
        print(f"{'PASS' if cond else 'FAIL'} {name} {info}", flush=True)

    try:
        if not gz.wait(lambda: link.m.target_system != 0, 180):
            raise RuntimeError("no MAVLink heartbeat from PX4")
        report["params_at_boot"] = {n: px4.cmd("param", "show", n).strip().splitlines()[-1] for n, _ in PARAMS[:3]}
        gz.sim_sleep(5)

        print(px4.cmd("ofnav", "start", "-p", "flow", "-q", args.min_quality, "-e"), flush=True)

        def armed():
            link.command(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 1)
            return gz.wait(lambda: link.state["armed"], 3)
        if not gz.wait(armed, 120, 1):
            raise RuntimeError("arming failed: " + px4.cmd("commander", "check"))
        link.command(mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, float("nan"),
                     float("nan"), float("nan"), float("nan"))
        if not gz.wait(lambda: link.state["z"] > 1.7, 60):
            raise RuntimeError(f"takeoff failed z={link.state['z']}")
        gz.sim_sleep(4)

        # Switch EKF2 from its own flow fusion to ofnav only.
        px4.cmd("param", "set", "EKF2_OF_CTRL", "0")
        gz.sim_sleep(5)
        flags = topic(px4, "estimator_status_flags")
        lp = topic(px4, "vehicle_local_position")
        status = px4.cmd("ofnav", "status")
        m = re.search(r"published=(\d+) skipped=(\d+)", status)
        published = int(m.group(1)) if m else 0
        check("1_ekf2_aided_only_by_ofnav",
              flags.get("cs_ev_vel") == "True" and flags.get("cs_opt_flow") == "False"
              and flags.get("cs_gnss_pos", flags.get("cs_gps")) == "False" and lp.get("xy_valid") == "True"
              and published > 20,
              cs_ev_vel=flags.get("cs_ev_vel"), cs_opt_flow=flags.get("cs_opt_flow"),
              cs_gnss_pos=flags.get("cs_gnss_pos", flags.get("cs_gps")), xy_valid=lp.get("xy_valid"),
              published=published, skipped=int(m.group(2)) if m else None)
        report["ofnav_status_aided"] = status

        link.set_mode(3)
        in_pos = gz.wait(lambda: link.mode() == "POSCTL", 10)
        gz.sim_sleep(5)
        t0 = link.boot_s
        truth0, est0 = xy(topic(px4, "vehicle_local_position_groundtruth")), xy(topic(px4, "vehicle_local_position"))
        drift, err = [], []
        while link.boot_s - t0 < HOVER_S:
            gz.sim_sleep(5)
            truth, est = xy(topic(px4, "vehicle_local_position_groundtruth")), xy(topic(px4, "vehicle_local_position"))
            d_truth = (truth[0] - truth0[0], truth[1] - truth0[1])
            d_est = (est[0] - est0[0], est[1] - est0[1])
            drift.append(math.hypot(*d_truth))
            err.append(math.hypot(d_est[0] - d_truth[0], d_est[1] - d_truth[1]))
            report["samples"].append({"t": round(link.boot_s - t0, 1), "truth_drift_m": round(drift[-1], 3),
                                      "estimate_error_m": round(err[-1], 3), "mode": link.mode(),
                                      "z": round(link.state["z"], 2)})
            print(report["samples"][-1], flush=True)
        check("2_position_hold_without_gnss", in_pos and link.mode() == "POSCTL" and drift
              and max(drift) < MAX_HOVER_DRIFT_M and max(err) < MAX_ESTIMATE_ERROR_M,
              hover_s=round(link.boot_s - t0, 1), max_true_drift_m=round(max(drift), 2) if drift else None,
              max_estimate_error_m=round(max(err), 2) if err else None)

        # Forward stick, release: stop and hold. Two inputs: inside the flow
        # profile envelope (<= 1.5 m/s) and beyond it, where ofnav rejects
        # measurements (FLOW_RATE_LIMIT) and EKF2 dead-reckons.
        for label, stick, seconds in (("inside_envelope", 300, 5), ("beyond_envelope", 600, 3)):
            truth_a, est_a = xy(topic(px4, "vehicle_local_position_groundtruth")), xy(topic(px4, "vehicle_local_position"))
            before = px4.cmd("ofnav", "status")
            link.pitch = stick
            t_start, vmax = link.boot_s, 0.0
            while link.boot_s - t_start < seconds:
                gt = topic(px4, "vehicle_local_position_groundtruth")
                vmax = max(vmax, math.hypot(float(gt.get("vx", 0.0)), float(gt.get("vy", 0.0))))
            link.pitch = 0
            gz.sim_sleep(10)
            truth_b, est_b = xy(topic(px4, "vehicle_local_position_groundtruth")), xy(topic(px4, "vehicle_local_position"))
            gz.sim_sleep(15)
            truth_c = xy(topic(px4, "vehicle_local_position_groundtruth"))
            moved_true = math.hypot(truth_b[0] - truth_a[0], truth_b[1] - truth_a[1])
            moved_est = math.hypot(est_b[0] - est_a[0], est_b[1] - est_a[1])
            hold_after = math.hypot(truth_c[0] - truth_b[0], truth_c[1] - truth_b[1])
            report[f"ofnav_status_before_{label}"] = before
            report[f"ofnav_status_after_{label}"] = px4.cmd("ofnav", "status")
            info = dict(stick=stick, max_true_speed_m_s=round(vmax, 2), moved_true_m=round(moved_true, 2),
                        moved_estimated_m=round(moved_est, 2), drift_after_stop_15s_m=round(hold_after, 2))
            if label == "inside_envelope":
                check("3_stick_move_stop_and_hold", moved_true > 1.0 and hold_after < 1.0
                      and abs(moved_est - moved_true) < 0.15 * moved_true + 0.3, **info)
            else:
                report["beyond_envelope_move"] = info  # documented, not a pass criterion
                print("INFO beyond_envelope", info, flush=True)

        link.set_mode(4, 6)
        landing = gz.wait(lambda: link.mode() == "AUTO.LAND", 10)
        landed = gz.wait(lambda: not link.state["armed"], 90)
        check("4_land_on_ofnav_estimates", landing and landed, disarmed=landed)
        report["final_ofnav_status"] = px4.cmd("ofnav", "status")
    except Exception as exc:  # report the failure, never hide it
        ok_all = False
        report["error"] = repr(exc)
        print("ERROR", exc, flush=True)
    finally:
        link.close()
        px4.stop()
    report["passed"] = ok_all
    (args.out / "gz_no_gnss_flight.json").write_text(json.dumps(report, indent=2))
    raise SystemExit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
