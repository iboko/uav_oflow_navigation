#!/usr/bin/env python3
"""Closed-loop PX4 SITL + Gazebo (gz_x500_flow) failure scenarios for ofnav.

The vehicle is armed and flown; ofnav runs with the optical-flow sensor
profile and opt-in safety actions (-a). Each scenario checks the whole
chain: navigation health -> SafetyActionManager -> vehicle_command ->
commander mode change -> vehicle behaviour.

  0  disarmed on the ground: no command may be sent
  1  short flow loss in hover (SITL input masking 'ofnav fault flow', 1.5 s)
       -> ALTITUDE_HOLD -> PX4 ALTCTL, altitude kept; restore -> NONE,
          the vehicle is NOT switched back to a position mode
  2  physical envelope: climb above 4 m with the 0.2-4 m flow profile
       -> HEIGHT_TOO_HIGH -> ALTITUDE_HOLD -> ALTCTL
  3  rangefinder loss (masking 'ofnav fault range')
       -> FAILSAFE_LAND -> LAND (latched) -> AUTO_LAND, landed, disarmed
  4  second flight: IMU invalid in ofnav (masking 'ofnav fault imu')
       -> FAILSAFE_LAND -> LAND (latched) -> AUTO_LAND, landed, disarmed;
          the manager was reset by the new arming (no carried-over LAND)

gz_bridge implements no 'failure optical_flow/distance_sensor' injection,
hence the module-boundary masking (POSIX builds only). Manual control is
a centred MAVLink MANUAL_CONTROL stream (pilot hands off sticks), required
by PX4 for ALTCTL. Writes a JSON report; exit code != 0 on any failure.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import threading
import time
from pathlib import Path

from pymavlink import mavutil

TEXTURED_WORLD = "ofnav_textured"

MAIN_MODE = {1: "MANUAL", 2: "ALTCTL", 3: "POSCTL", 4: "AUTO", 7: "STABILIZED"}
AUTO_SUB = {2: "TAKEOFF", 3: "LOITER", 6: "LAND"}


def install_textured_world(px4_dir: Path, seed: int = 7) -> None:
    """Writes <PX4>/Tools/simulation/gz/worlds/ofnav_textured.sdf.

    The stock worlds give the 100x100 px, 42 deg flow camera (PX4 OpenCV
    flow) a quality of 0-60 out of 255 under software rendering: a uniform
    plane (default), sparse markers (aruco) or a heavy terrain mesh with a
    low frame rate (baylands). This world is the stock 'default' world with
    a 20 x 20 m ground texture of multi-scale random blobs (1 cm/px, i.e.
    ~2 cm/px seen from 2.5 m), generated deterministically from 'seed'.
    """
    import numpy as np
    from PIL import Image

    worlds = px4_dir / "Tools" / "simulation" / "gz" / "worlds"
    texture = worlds / f"{TEXTURED_WORLD}.png"
    rng = np.random.default_rng(seed)
    n = 2048
    img = np.zeros((n, n), np.float32)
    for cells, weight in ((16, 0.6), (64, 1.0), (256, 0.8), (1024, 0.5)):
        coarse = rng.random((cells, cells)).astype(np.float32)
        img += weight * np.asarray(Image.fromarray(coarse).resize((n, n), Image.BICUBIC))
    img = (img - img.min()) / (img.max() - img.min())
    Image.fromarray((255 * img).astype(np.uint8)).convert("RGB").save(texture)
    sdf = (worlds / "default.sdf").read_text(encoding="utf-8-sig")
    sdf = sdf.replace('<world name="default">', f'<world name="{TEXTURED_WORLD}">', 1)
    stock = sdf[sdf.index("<visual name=\"visual\">"):]
    stock = stock[:stock.index("</visual>") + len("</visual>")]
    sdf = sdf.replace(stock, f"""<visual name="visual">
          <geometry><plane><normal>0 0 1</normal><size>20 20</size></plane></geometry>
          <material>
            <ambient>1 1 1 1</ambient><diffuse>1 1 1 1</diffuse><specular>0 0 0 1</specular>
            <pbr><metal><albedo_map>{texture}</albedo_map><roughness>1</roughness></metal></pbr>
          </material>
        </visual>""", 1)
    (worlds / f"{TEXTURED_WORLD}.sdf").write_text(sdf, encoding="utf-8")


def _gz_server_pids() -> set[int]:
    """PIDs of running 'gz sim' processes (Linux /proc)."""
    pids = set()
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            argv = (d / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if any(a.endswith(b"gz") for a in argv[:2]) and b"sim" in argv[1:3]:
            pids.add(int(d.name))
    return pids


class Px4:
    def __init__(self, px4_dir: Path, log: Path, world: str = "default",
                 params: dict[str, str] | None = None, model: str = "gz_x500_flow"):
        self.build = px4_dir / "build" / "px4_sitl_default"
        self.rootfs = self.build / "rootfs"
        env = dict(os.environ, HEADLESS="1", PX4_SIM_MODEL=model, GZ_IP="127.0.0.1",
                   PX4_GZ_WORLD=world)
        # Boot-time parameters (rcS: PX4_PARAM_<name>), set before modules start.
        env.update({f"PX4_PARAM_{k}": v for k, v in (params or {}).items()})
        self.log = log.open("w")
        self._gz_before = _gz_server_pids()
        # Isolate the SITL/Gazebo process tree. Never terminate unrelated
        # Gazebo simulations running on the same development machine.
        self.proc = subprocess.Popen([str(self.build / "bin" / "px4"), "-d"], cwd=self.rootfs,
                                     env=env, stdout=self.log, stderr=subprocess.STDOUT,
                                     start_new_session=True)

    def cmd(self, *args: str, timeout: float = 120.) -> str:
        r = subprocess.run([str(self.build / "bin" / f"px4-{args[0]}"), *args[1:]],
                           cwd=self.rootfs, capture_output=True, text=True, timeout=timeout)
        return r.stdout + r.stderr

    def stop(self) -> None:
        try:
            self.cmd("shutdown", timeout=10)
        except Exception:
            pass
        # Own session = own process group, including Gazebo children which
        # may outlive the PX4 parent. A global 'pkill gz sim' is unsafe.
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.proc.wait(timeout=10)
        finally:
            # The Gazebo server can leave PX4's process group and outlive it;
            # the next start then attached to that stale world ("gazebo
            # already running world"). Stop only the servers started since
            # this instance began, never pre-existing ones.
            for pid in _gz_server_pids() - self._gz_before:
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            deadline = time.time() + 10
            while (_gz_server_pids() - self._gz_before) and time.time() < deadline:
                time.sleep(0.2)
            for pid in _gz_server_pids() - self._gz_before:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            self.log.close()


class Link:
    """GCS heartbeat + centred manual control, telemetry cache."""

    def __init__(self):
        self.m = mavutil.mavlink_connection("udpin:0.0.0.0:14550", source_system=255)
        self.state = {"armed": False, "main": 0, "sub": 0, "z": 0., "landed": 0}
        self.boot_s = 0.  # PX4 (lockstep = simulation) time from MAVLink
        self.throttle = 500
        self.pitch = 0  # MANUAL_CONTROL x: + forward
        self.roll = 0   # MANUAL_CONTROL y: + right
        self._stop = False
        threading.Thread(target=self._tx, daemon=True).start()
        threading.Thread(target=self._rx, daemon=True).start()

    def _tx(self) -> None:
        last_hb = 0.
        while not self._stop:
            now = time.time()
            if self.m.target_system:
                if now - last_hb > 1.:
                    self.m.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS,
                                              mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
                    last_hb = now
                self.m.mav.manual_control_send(self.m.target_system, self.pitch, self.roll, self.throttle, 0, 0)
            time.sleep(0.05)

    def _rx(self) -> None:
        while not self._stop:
            msg = self.m.recv_match(blocking=True, timeout=1)
            if msg is None:
                continue
            t = msg.get_type()
            boot_ms = getattr(msg, "time_boot_ms", None)
            if boot_ms is not None and msg.get_srcComponent() == 1:
                self.boot_s = max(self.boot_s, boot_ms * 1e-3)
            if t == "HEARTBEAT" and msg.get_srcComponent() == 1 and msg.type != 6:
                self.state["armed"] = bool(msg.base_mode & 128)
                self.state["main"] = (msg.custom_mode >> 16) & 0xFF
                self.state["sub"] = (msg.custom_mode >> 24) & 0xFF
            elif t == "LOCAL_POSITION_NED":
                self.state["z"] = -msg.z
            elif t == "EXTENDED_SYS_STATE":
                self.state["landed"] = msg.landed_state

    def mode(self) -> str:
        main, sub = self.state["main"], self.state["sub"]
        name = MAIN_MODE.get(main, str(main))
        return f"{name}.{AUTO_SUB.get(sub, sub)}" if main == 4 else name

    def command(self, cmd: int, *params: float) -> None:
        p = list(params) + [0.] * (7 - len(params))
        self.m.mav.command_long_send(self.m.target_system, 1, cmd, 0, *p)

    def set_mode(self, main: int, sub: int = 0) -> None:
        self.command(mavutil.mavlink.MAV_CMD_DO_SET_MODE, 1, main, sub)

    def close(self) -> None:
        self._stop = True


SIM_CLOCK: "Link | None" = None


def wait(pred, timeout: float, step: float = 0.2) -> bool:
    """Waits up to 'timeout' seconds of SIMULATION time (PX4 lockstep).

    Software-rendered Gazebo runs at a real-time factor of 0.1-1 depending
    on the machine, so wall-clock timeouts would make the scenarios depend
    on the host. Wall-clock time is used only before the first MAVLink
    timestamp, and as a hard cap (simulation stalled).
    """
    sim = SIM_CLOCK if SIM_CLOCK and SIM_CLOCK.boot_s > 0. else None

    def now() -> float:
        return sim.boot_s if sim else time.time()

    start, wall_cap = now(), time.time() + 60. * timeout + 60.
    while now() - start <= timeout and time.time() < wall_cap:
        if pred():
            return True
        time.sleep(step)
    return bool(pred())


def sim_sleep(seconds: float) -> None:
    wait(lambda: False, seconds)


def ofnav_status(px4: Px4) -> dict:
    text = px4.cmd("ofnav", "status")
    out = {"raw": text}
    m = re.search(r"safety action=(\S+) reason=(\S+) commands=(\S+) sent=(\d+)", text)
    if m:
        out.update(action=m.group(1), reason=m.group(2), sent=int(m.group(4)))
    m = re.search(r"samples=(\d+) accepted=(\d+) degraded=(\d+) failsafe=(\d+)", text)
    if m:
        out.update(samples=int(m.group(1)), accepted=int(m.group(2)),
                   degraded=int(m.group(3)), failsafe=int(m.group(4)))
    return out


def new_report(world: str) -> dict:
    """Provenance and explicit non-claims for safety-only SITL tests."""
    return {
        "type": "PX4_SITL_GAZEBO_X500_FLOW_FAILURE_SCENARIOS",
        "world": world,
        "scenarios": {},
        "validation_scope": "PX4_SITL_SAFETY_MODE_TRANSITIONS_ONLY",
        "gnss_denied_navigation_verified": False,
        "ekf2_ofnav_fusion_verified": False,
        "ofnav_position_hold_verified": False,
        "real_aircraft_tested": False,
        "caveat": ("POSCTL is provided by PX4; scenario success proves the safety "
                   "mode-transition chain, not GNSS-denied ofnav position estimation."),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("px4_dir", type=Path)
    ap.add_argument("--out", type=Path, default=Path("build/gz_failure_scenarios"))
    ap.add_argument("--world", default=TEXTURED_WORLD,
                    help="gz world (default: generated textured ground, see install_textured_world)")
    ap.add_argument("--min-quality", default="60", help="ofnav -q for the simulated flow sensor")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.world == TEXTURED_WORLD:
        install_textured_world(args.px4_dir.resolve())
    report: dict = new_report(args.world)
    # Set at boot (PX4_PARAM_*). COM_RC_OVERRIDE=0: the harness moves the
    # throttle stick between scenarios; commander must not read that as the
    # pilot taking over an ofnav-commanded AUTO.LAND.
    px4 = Px4(args.px4_dir.resolve(), args.out / "px4_gz.log", args.world,
              {"MIS_TAKEOFF_ALT": "2.0", "NAV_DLL_ACT": "0", "COM_RC_IN_MODE": "1",
               "NAV_RCL_ACT": "0", "COM_DISARM_LAND": "2", "COM_RC_OVERRIDE": "0"})
    link = Link()
    global SIM_CLOCK
    SIM_CLOCK = link
    ok_all = True

    def check(name: str, cond: bool, **info) -> None:
        nonlocal ok_all
        ok_all = ok_all and bool(cond)
        report["scenarios"][name] = {"pass": bool(cond), **info}
        print(f"{'PASS' if cond else 'FAIL'} {name} {info}", flush=True)

    try:
        if not wait(lambda: link.m.target_system != 0, 180):
            raise RuntimeError("no MAVLink heartbeat from PX4")
        print(px4.cmd("ofnav", "start", "-p", "flow", "-a", "-q", args.min_quality), flush=True)
        sim_sleep(5)
        s0 = ofnav_status(px4)
        check("0_disarmed_no_command", s0.get("sent") == 0, action=s0.get("action"))

        def armed():
            link.command(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 1)
            return wait(lambda: link.state["armed"], 3)
        if not wait(armed, 180, 1):
            raise RuntimeError("arming failed: " + px4.cmd("commander", "check"))
        link.command(mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, float("nan"),
                     float("nan"), float("nan"), float("nan"))
        if not wait(lambda: link.state["z"] > 1.7, 60):
            raise RuntimeError(f"takeoff failed z={link.state['z']}")
        sim_sleep(4)
        link.set_mode(3)  # pilot takes position mode, sticks centred
        wait(lambda: link.mode() == "POSCTL", 10)
        sim_sleep(4)
        hover = ofnav_status(px4)
        check("hover_flow_nav", hover.get("action") == "NONE" and hover.get("accepted", 0) > 20,
              mode=link.mode(), z=round(link.state["z"], 2), accepted=hover.get("accepted"))
        report["hover_ofnav_status"] = hover.get("raw")

        # 1: short flow loss -> ALTITUDE_HOLD -> ALTCTL; restore -> NONE,
        # stays ALTCTL. Loss + 1 s recovery must stay below the 5 s
        # escalation to LAND (SafetyActionConfig::degraded_to_land_us).
        z_before = link.state["z"]
        t_fault = link.boot_s
        px4.cmd("ofnav", "fault", "flow")
        got = wait(lambda: link.mode() == "ALTCTL", 10)
        s1 = ofnav_status(px4)
        sim_sleep(1.5)
        z_after = link.state["z"]
        px4.cmd("ofnav", "fault", "none")
        t_restore = link.boot_s
        loss_s = round(t_restore - t_fault, 1)
        wait(lambda: ofnav_status(px4).get("action") == "NONE", 15, 0.5)
        recovery_s = round(link.boot_s - t_restore, 1)  # simulation time
        s1b = ofnav_status(px4)
        report["fault_flow_ofnav_status"] = s1.get("raw")
        check("1_flow_loss_altitude_hold", got and s1.get("action") == "ALTITUDE_HOLD"
              and abs(z_after - z_before) < 0.6 and s1b.get("action") == "NONE"
              and link.mode() == "ALTCTL",
              mode=link.mode(), sent=s1b.get("sent"), dz=round(z_after - z_before, 2),
              recovered=s1b.get("action"), loss_s=loss_s, recovery_s=recovery_s)
        report["recovered_ofnav_status"] = s1b.get("raw")

        # 2: climb above the 4 m flow envelope in POSCTL
        link.set_mode(3)
        wait(lambda: link.mode() == "POSCTL", 10)
        sim_sleep(2)
        link.throttle = 750
        reached = wait(lambda: link.state["z"] > 4.0, 40)
        link.throttle = 500
        got = wait(lambda: link.mode() == "ALTCTL", 10)
        s2 = ofnav_status(px4)
        report["height_ofnav_status"] = s2.get("raw")
        check("2_height_envelope_altitude_hold", reached and got and s2.get("action") == "ALTITUDE_HOLD",
              z=round(link.state["z"], 2), mode=link.mode(), reason=s2.get("reason"), sent=s2.get("sent"))
        link.throttle = 300
        wait(lambda: link.state["z"] < 2.3, 40)
        link.throttle = 500
        sim_sleep(3)

        # 3: rangefinder loss -> LAND (latched) -> AUTO.LAND, landed, disarmed
        px4.cmd("ofnav", "fault", "range")
        got = wait(lambda: link.mode() == "AUTO.LAND", 10)
        s3 = ofnav_status(px4)
        landed = wait(lambda: not link.state["armed"], 90)
        px4.cmd("ofnav", "fault", "none")
        sim_sleep(2)
        s3b = ofnav_status(px4)
        check("3_range_loss_land", got and landed and s3.get("action") == "LAND"
              and s3b.get("action") == "LAND",
              mode_seen="AUTO.LAND" if got else link.mode(), disarmed=landed,
              latched=s3b.get("action"), sent=s3b.get("sent"))
        report["final_ofnav_status"] = s3b.get("raw")

        # 4: second flight, IMU loss in ofnav -> LAND
        if not wait(armed, 120, 1):
            raise RuntimeError("re-arming failed: " + px4.cmd("commander", "check"))
        link.command(mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, float("nan"),
                     float("nan"), float("nan"), float("nan"))
        if not wait(lambda: link.state["z"] > 1.7, 60):
            raise RuntimeError(f"second takeoff failed z={link.state['z']}")
        sim_sleep(4)
        link.set_mode(3)
        wait(lambda: link.mode() == "POSCTL", 10)
        engaged = wait(lambda: "safety engaged=1" in ofnav_status(px4).get("raw", ""), 20, 0.5)
        s4a = ofnav_status(px4)  # the LAND of flight 1 must not carry over
        px4.cmd("ofnav", "fault", "imu")
        got = wait(lambda: link.mode() == "AUTO.LAND", 10)
        s4 = ofnav_status(px4)
        landed = wait(lambda: not link.state["armed"], 90)
        px4.cmd("ofnav", "fault", "none")
        sim_sleep(2)
        s4b = ofnav_status(px4)
        report["imu_fault_ofnav_status"] = s4.get("raw")
        check("4_imu_loss_land", engaged and s4a.get("action") == "NONE" and got and landed
              and s4.get("action") == "LAND" and s4b.get("action") == "LAND",
              action_after_rearm=s4a.get("action"), engaged=engaged,
              mode_seen="AUTO.LAND" if got else link.mode(), disarmed=landed,
              reason=s4.get("reason"), latched=s4b.get("action"))
    except Exception as exc:  # report the failure, never hide it
        ok_all = False
        report["error"] = repr(exc)
        print("ERROR", exc, flush=True)
    finally:
        link.close()
        px4.stop()
    report["passed"] = ok_all
    (args.out / "gz_failure_scenarios.json").write_text(json.dumps(report, indent=2))
    raise SystemExit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
