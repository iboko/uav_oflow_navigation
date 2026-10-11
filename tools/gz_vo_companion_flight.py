#!/usr/bin/env python3
"""GNSS loss in flight, EKF2 continues on companion-computer VO: PX4 SITL + Gazebo.

Vehicle: x500 with a downward camera and a downward lidar (model and airframe
installed by this script). Companion process: tools/gz_vo_companion.py
(Gazebo camera -> VO -> MAVLink ODOMETRY body-FRD velocity, TIMESYNC both
ways). EKF2 fuses GNSS and the VO velocity (EKF2_EV_CTRL=4) from takeoff.

  1  with GNSS: EKF2 fuses the companion velocity (cs_ev_vel) next to
     GNSS (cs_gnss_pos); VO velocity vs ground truth over a hover and a
     slow forward movement (bias, RMSE)
  2  GNSS lost in flight. Gazebo has no GNSS failure injection, so the
     loss is applied in the estimator: EKF2_GPS_CTRL=0. From then on EKF2's
     only horizontal aiding is the companion VO: cs_ev_vel, not
     cs_gnss_pos, horizontal position valid
  3  POSCTL hover 60 s: true drift and estimate error
  4  stick forward inside the VO envelope, release: estimated vs true
     displacement, stop and hold
  5  AUTO.LAND on VO: landed and disarmed

Ground truth: vehicle_local_position_groundtruth (gz_bridge). All waits
in simulation time. Writes a JSON report; exit code != 0 on any failure.
"""
from __future__ import annotations

import argparse
import json
import math
import signal
import subprocess
import sys
from pathlib import Path

from pymavlink import mavutil

import gz_failure_scenarios as gz
from gz_no_gnss_flight import topic, xy

MODEL = "x500_vo_down"
AIRFRAME = f"4901_gz_{MODEL}"
CAMERA_TOPIC = "/ofnav/vo_camera"
PARAMS = {"EKF2_EV_CTRL": "4", "EKF2_EV_DELAY": "0", "EKF2_HGT_REF": "0",
          "NAV_DLL_ACT": "0", "COM_RC_IN_MODE": "1", "NAV_RCL_ACT": "0", "COM_RC_OVERRIDE": "0",
          "COM_DISARM_LAND": "2", "MIS_TAKEOFF_ALT": "3.0"}
HOVER_S = 60.0


def install_vehicle(px4_dir: Path, width: int = 320, height: int = 240, hfov: float = 1.2,
                    rate_hz: int = 15) -> None:
    """x500 + downward camera (small image: software rendering) + the
    downward lidar of x500_flow (gz_bridge reads it as distance_sensor)."""
    models = px4_dir / "Tools" / "simulation" / "gz" / "models"
    flow = (models / "x500_flow" / "model.sdf").read_text()
    lidar = flow[flow.index('<joint name="lidar_model_joint"'):flow.rindex("</model>")]
    sdf = f"""<?xml version="1.0" encoding="UTF-8"?>
<sdf version='1.9'>
  <model name='{MODEL}'>
    <self_collide>false</self_collide>
    <include merge='true'><uri>x500</uri></include>
    <include merge='true'>
      <uri>model://LW20</uri>
      <pose relative_to="base_link">0 0 -0.079 0 1.57 0</pose>
    </include>
    <link name="vo_camera_link">
      <pose relative_to="base_link">0 0 -0.1 0 1.5707963 0</pose>
      <inertial><mass>0.001</mass><inertia><ixx>1e-5</ixx><iyy>1e-5</iyy><izz>1e-5</izz>
        <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>
      <sensor name="vo_camera" type="camera">
        <topic>{CAMERA_TOPIC}</topic>
        <camera>
          <horizontal_fov>{hfov}</horizontal_fov>
          <image><width>{width}</width><height>{height}</height><format>L8</format></image>
          <clip><near>0.05</near><far>100</far></clip>
        </camera>
        <always_on>1</always_on>
        <update_rate>{rate_hz}</update_rate>
      </sensor>
    </link>
    <joint name="vo_camera_joint" type="fixed">
      <parent>base_link</parent><child>vo_camera_link</child>
    </joint>
    {lidar}
  </model>
</sdf>
"""
    (models / MODEL).mkdir(exist_ok=True)
    (models / MODEL / "model.sdf").write_text(sdf)
    (models / MODEL / "model.config").write_text(
        f'<?xml version="1.0"?>\n<model><name>{MODEL}</name><version>1.0</version>'
        f'<sdf version="1.9">model.sdf</sdf><description>x500, downward camera and lidar for '
        f'companion VO (ofnav)</description></model>\n')
    airframe = (f"#!/bin/sh\n#\n# @name Gazebo x500 with downward camera and lidar (ofnav companion VO)\n"
                f"#\n# @type Quadrotor\n#\n\nPX4_SIM_MODEL=${{PX4_SIM_MODEL:={MODEL}}}\n\n"
                f". ${{R}}etc/init.d-posix/airframes/4001_gz_x500\n")
    for d in (px4_dir / "ROMFS" / "px4fmu_common" / "init.d-posix" / "airframes",
              px4_dir / "build" / "px4_sitl_default" / "etc" / "init.d-posix" / "airframes"):
        if d.is_dir():
            (d / AIRFRAME).write_text(airframe)


def flags_and_validity(px4: gz.Px4) -> dict:
    flags = topic(px4, "estimator_status_flags")
    lp = topic(px4, "vehicle_local_position")
    return {"cs_ev_vel": flags.get("cs_ev_vel"), "cs_gnss_pos": flags.get("cs_gnss_pos", flags.get("cs_gps")),
            "cs_opt_flow": flags.get("cs_opt_flow"), "xy_valid": lp.get("xy_valid")}


def wait_estimator_settled(px4: gz.Px4, timeout_s: float = 180.0, max_heading_error_deg: float = 10.0) -> dict:
    """Preflight gate from EKF2's own signals only (never ground truth).

    5 consecutive seconds: GNSS fused, magnetometer fused (heading or 3D) and
    consistent, at rest, heading steady within 1 deg. Under load the SITL
    start can leave EKF2 unaided for ~30 s, or without magnetometer fusion
    and a 45-50 deg heading error, while the arming checks already pass;
    after takeoff the controller then turns the vehicle by tens of degrees
    (seen in the ulogs). A consistent magnetometer heading can still be
    5-10 deg off in SITL, as on a real vehicle.
    """
    state = {"n": 0, "yaw": None}

    def settled() -> bool:
        f = topic(px4, "estimator_status_flags")
        lp = topic(px4, "vehicle_local_position")
        yaw = float(lp.get("heading", math.nan))
        mag = f.get("cs_mag_3d") == "True" or f.get("cs_mag_hdg") == "True"
        still = (f.get("cs_gnss_pos", f.get("cs_gps")) == "True" and mag
                 and f.get("cs_mag_heading_consistent") == "True"
                 and all(abs(float(lp.get(k, 9.0))) < 0.1 for k in ("vx", "vy", "vz"))
                 and state["yaw"] is not None
                 and abs((yaw - state["yaw"] + math.pi) % (2 * math.pi) - math.pi) < math.radians(1.0))
        state["yaw"] = yaw
        state["n"] = state["n"] + 1 if still else 0
        return state["n"] >= 5
    if not gz.wait(settled, timeout_s, 1):
        raise SitlStartRejected("EKF2 not settled on the ground (GNSS, magnetometer, rest, steady heading): "
                                + json.dumps(flags_and_validity(px4)))
    # Simulation sanity, not navigation: some SITL starts align the heading
    # 30-50 deg off while every EKF2 flag reads healthy (the heading jumps when
    # GNSS fusion starts). Such a start is discarded and counted; ground truth
    # never reaches the companion or the estimator. 10 deg leaves realistic
    # magnetometer heading errors in the test.
    truth = float(topic(px4, "vehicle_local_position_groundtruth").get("heading", math.nan))
    err = math.degrees(abs((state["yaw"] - truth + math.pi) % (2 * math.pi) - math.pi))
    if not err < max_heading_error_deg:
        raise SitlStartRejected(f"SITL start rejected: EKF2 heading {err:.1f} deg from simulator truth")
    return {"heading_rad": state["yaw"], "heading_error_vs_truth_deg": round(err, 2)}


class SitlStartRejected(RuntimeError):
    """The simulated start is unusable (not a navigation failure)."""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("px4_dir", type=Path)
    ap.add_argument("--out", type=Path, default=Path("build/gz_vo_companion"))
    ap.add_argument("--world", default=gz.TEXTURED_WORLD)
    ap.add_argument("--companion-python", default=sys.executable,
                    help="interpreter with gz.transport13 / gz.msgs10, numpy, cv2, pymavlink")
    args = ap.parse_args()
    # SIGTERM (timeout) must still stop PX4, Gazebo and the companion.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))
    args.out.mkdir(parents=True, exist_ok=True)
    px4_dir = args.px4_dir.resolve()
    if args.world == gz.TEXTURED_WORLD:
        gz.install_textured_world(px4_dir)
    install_vehicle(px4_dir)
    report: dict = {"type": "PX4_SITL_GAZEBO_GNSS_LOSS_COMPANION_VO", "world": args.world,
                    "params": PARAMS, "checks": {}, "samples": []}
    px4 = gz.Px4(px4_dir, args.out / "px4_gz.log", args.world, PARAMS, model=f"gz_{MODEL}")
    link = gz.Link()
    gz.SIM_CLOCK = link
    companion_log = (args.out / "companion.log").open("w")
    companion = None
    ok_all = True

    def check(name: str, cond: bool, **info) -> None:
        nonlocal ok_all
        ok_all = ok_all and bool(cond)
        report["checks"][name] = {"pass": bool(cond), **info}
        print(f"{'PASS' if cond else 'FAIL'} {name} {info}", flush=True)

    def truth_est():
        return xy(topic(px4, "vehicle_local_position_groundtruth")), xy(topic(px4, "vehicle_local_position"))

    try:
        if not gz.wait(lambda: link.m.target_system != 0, 180):
            raise RuntimeError("no MAVLink heartbeat from PX4")
        # Only now: PX4's startup takes any /world/*/clock topic, even a mere
        # subscription, for an already running Gazebo world.
        companion = subprocess.Popen(
            [args.companion_python, str(Path(__file__).with_name("gz_vo_companion.py")), "--world", args.world,
             "--camera-topic", CAMERA_TOPIC, "--log", str(args.out / "companion_vo.csv")],
            stdout=companion_log, stderr=subprocess.STDOUT)
        # On the ground the camera is ~0.2 m above the texture and VO has no
        # track: the takeoff is GNSS-aided, VO fusion is checked in the air.
        # Preflight: EKF2 aided by GNSS and at rest for 5 s. Under load the
        # SITL start can leave EKF2 unaided for ~30 s (hundreds of metres of
        # drift) and still converging when the arming checks already pass.
        report["preflight"] = wait_estimator_settled(px4)

        def armed():
            link.command(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 1)
            return gz.wait(lambda: link.state["armed"], 3)
        if not gz.wait(armed, 180, 1):
            raise RuntimeError("arming failed: " + px4.cmd("commander", "check"))
        z_home = link.state["z"]
        link.command(mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, float("nan"),
                     float("nan"), float("nan"), float("nan"))
        if not gz.wait(lambda: link.state["z"] - z_home > 2.7, 60):
            raise RuntimeError(f"takeoff failed dz={link.state['z'] - z_home}")
        gz.sim_sleep(4)
        link.set_mode(3)
        gz.wait(lambda: link.mode() == "POSCTL", 10)
        gz.sim_sleep(5)

        # 1: both sources fused; VO accuracy against ground truth while GNSS holds.
        f1 = flags_and_validity(px4)
        sync = topic(px4, "timesync_status")
        report["px4_timesync_with_companion"] = {k: sync.get(k) for k in
                                                ("source_protocol", "remote_timestamp", "observed_offset",
                                                 "estimated_offset", "round_trip_time")}
        errors = []
        for stick, seconds in ((0, 8), (250, 6), (0, 8)):
            link.pitch = stick
            t_start = link.boot_s
            while link.boot_s - t_start < seconds:
                gt = topic(px4, "vehicle_local_position_groundtruth")
                vo = topic(px4, "vehicle_visual_odometry")
                att = topic(px4, "vehicle_attitude")
                if not all(k in vo for k in ("velocity[0]", "velocity[1]")) or "q[0]" not in att:
                    continue
                q = [float(att[f"q[{i}]"]) for i in range(4)]
                yaw = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))
                bx, by = float(vo["velocity[0]"]), float(vo["velocity[1]"])
                vn, ve = math.cos(yaw) * bx - math.sin(yaw) * by, math.sin(yaw) * bx + math.cos(yaw) * by
                errors.append((vn - float(gt["vx"]), ve - float(gt["vy"]), math.hypot(gt["vx"], gt["vy"])))
        link.pitch = 0
        gz.sim_sleep(8)
        n = len(errors)
        rmse = math.sqrt(sum(e[0] ** 2 + e[1] ** 2 for e in errors) / n) if n else math.nan
        bias = (sum(e[0] for e in errors) / n, sum(e[1] for e in errors) / n) if n else (math.nan, math.nan)
        check("1_ekf2_fuses_companion_vo_with_gnss", f1["cs_ev_vel"] == "True" and f1["cs_gnss_pos"] == "True"
              and n > 10 and rmse < 0.3, **f1, samples=n, vo_velocity_rmse_m_s=round(rmse, 3),
              vo_velocity_bias_ne_m_s=[round(b, 3) for b in bias],
              max_true_speed_m_s=round(max((e[2] for e in errors), default=0.0), 2))

        # 2: GNSS lost in the estimator.
        px4.cmd("param", "set", "EKF2_GPS_CTRL", "0")
        gz.sim_sleep(5)
        f2 = flags_and_validity(px4)
        check("2_gnss_lost_ekf2_on_companion_vo_only", f2["cs_ev_vel"] == "True" and f2["cs_gnss_pos"] == "False"
              and f2["xy_valid"] == "True", **f2)

        # 3: hover.
        t0 = link.boot_s
        truth0, est0 = truth_est()
        drift, err = [], []
        while link.boot_s - t0 < HOVER_S:
            gz.sim_sleep(5)
            truth, est = truth_est()
            d_truth = (truth[0] - truth0[0], truth[1] - truth0[1])
            d_est = (est[0] - est0[0], est[1] - est0[1])
            drift.append(math.hypot(*d_truth))
            err.append(math.hypot(d_est[0] - d_truth[0], d_est[1] - d_truth[1]))
            report["samples"].append({"t": round(link.boot_s - t0, 1), "truth_drift_m": round(drift[-1], 3),
                                      "estimate_error_m": round(err[-1], 3), "mode": link.mode(),
                                      "z": round(link.state["z"], 2)})
            print(report["samples"][-1], flush=True)
        check("3_position_hold_on_companion_vo", link.mode() == "POSCTL" and drift
              and max(drift) < 2.0 and max(err) < 2.0,
              hover_s=round(link.boot_s - t0, 1), max_true_drift_m=round(max(drift), 2) if drift else None,
              max_estimate_error_m=round(max(err), 2) if err else None)

        # 4: stick forward, release.
        truth_a, est_a = truth_est()
        link.pitch = 300
        t_start, vmax = link.boot_s, 0.0
        while link.boot_s - t_start < 5:
            gt = topic(px4, "vehicle_local_position_groundtruth")
            vmax = max(vmax, math.hypot(float(gt.get("vx", 0.0)), float(gt.get("vy", 0.0))))
        link.pitch = 0
        gz.sim_sleep(10)
        truth_b, est_b = truth_est()
        gz.sim_sleep(15)
        truth_c, _ = truth_est()
        moved_true = math.hypot(truth_b[0] - truth_a[0], truth_b[1] - truth_a[1])
        moved_est = math.hypot(est_b[0] - est_a[0], est_b[1] - est_a[1])
        hold_after = math.hypot(truth_c[0] - truth_b[0], truth_c[1] - truth_b[1])
        check("4_stick_move_stop_and_hold_on_vo", moved_true > 1.0 and hold_after < 1.0
              and abs(moved_est - moved_true) < 0.15 * moved_true + 0.3,
              max_true_speed_m_s=round(vmax, 2), moved_true_m=round(moved_true, 2),
              moved_estimated_m=round(moved_est, 2), drift_after_stop_15s_m=round(hold_after, 2))

        # 5: land.
        link.set_mode(4, 6)
        landing = gz.wait(lambda: link.mode() == "AUTO.LAND", 10)
        landed = gz.wait(lambda: not link.state["armed"], 120)
        check("5_land_on_companion_vo", landing and landed, disarmed=landed)
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
        companion_log.close()
        px4.stop()
    log_text = (args.out / "companion.log").read_text()
    status = [json.loads(line) for line in log_text.splitlines() if line.startswith("{\"sim_s\"")]
    report["companion_final_status"] = status[-1] if status else None
    report["passed"] = ok_all
    (args.out / "gz_vo_companion_flight.json").write_text(json.dumps(report, indent=2))
    raise SystemExit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
