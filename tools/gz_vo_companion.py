#!/usr/bin/env python3
"""Companion-computer VO node: Gazebo camera -> VO -> MAVLink ODOMETRY -> EKF2.

A separate process, as on a companion computer. Its own clock is the
simulation clock (Gazebo /clock); the autopilot clock is reached through
MAVLink TIMESYNC in both directions:
  * the node answers PX4's TIMESYNC requests, so PX4 converts ODOMETRY
    time_usec (companion clock) into its own time base;
  * the node sends its own requests to convert the autopilot ATTITUDE_QUATERNION
    and DISTANCE_SENSOR stamps into the camera time base.

Inputs from the autopilot are the vehicle's own estimates and sensors
(attitude, rangefinder), never Gazebo ground truth. Each accepted image
pair becomes one ODOMETRY message: body-FRD velocity at the pair midpoint
(src/companion_vo.py), position and attitude NaN.

Needs the Gazebo Python bindings (gz.transport13, gz.msgs10; system Python
3.12 on Ubuntu 24.04) plus numpy, opencv-python-headless and pymavlink.
Prints a status JSON line every --status-s seconds.
"""
from __future__ import annotations

import argparse
import json
import math
import queue
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from gz.msgs10.clock_pb2 import Clock
from gz.msgs10.image_pb2 import Image
from gz.transport13 import Node
from pymavlink import mavutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Strong gusts, diag, seeds 33-36 and 41-44: all checks 7 of 8 flights with the
# bridge, 3 of 8 without (one relocked on the shifted map: 32 fixes fused).
DEFAULT_GAP_BRIDGE = Path(__file__).resolve().parents[1] / "calibration" / "vo_gap_bridge_swiss3d_50m_gusts.json"
from src.camera_rotation_compensation import CameraMountCalibration  # noqa: E402
from src.attitude_time_alignment import quaternion_to_rpy  # noqa: E402
from src.companion_map import GyroHeading, MapAnchor, VelocityBridge  # noqa: E402
from src.camera_rotation_compensation import body_to_ned  # noqa: E402
from src.companion_msckf import CompanionMsckf  # noqa: E402
from src.map_error_calibration import covariance_at_height, quality_covariance  # noqa: E402
from src.map_error_field import MapErrorField  # noqa: E402
from src.companion_vo import BodyVelocityMeasurement, CompanionVo, VelocityJumpGate, VerticalVelocityGate  # noqa: E402
from src.error_state_inertial_filter import ImuNoiseDensity  # noqa: E402
from src.feature_tracker import KltTracker  # noqa: E402
from src.imu_preintegration import ImuReading  # noqa: E402
from src.msckf_vio import PinholeCamera  # noqa: E402
from src.continuous_visual_odometry import ContinuousPlanarOdometry, VisualOdomResult  # noqa: E402
from src.surface_depth import SurfaceDepth  # noqa: E402
from src.ground_visual_motion import CameraCalibration  # noqa: E402
from src.ortho_ncc_localizer import OrthoNccLocalizer  # noqa: E402
from src.orthophoto_localization import MapGeoReference, OrthophotoLocalizer, OrthophotoTile  # noqa: E402
from src.velocity_uncertainty import VelocityErrorAssumptions  # noqa: E402

# Camera pitched +90 deg about body y (gz model): image right = body right,
# image down = body backward. Optical axis = body down.
IMAGE_TO_BODY = ((0.0, -1.0), (1.0, 0.0))
BODY_FROM_CAMERA = ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))


def calibration(width: int, height: int, hfov_rad: float, focal_error: float = 0.0) -> CameraCalibration:
    # focal_error: deliberate calibration error (fault injection), e.g. 0.05
    # gives a 5 % VO scale error, like a wrong focal length or height bias.
    f = 0.5 * width / math.tan(0.5 * hfov_rad) * (1.0 + focal_error)
    return CameraCalibration(f, f, 0.5 * width, 0.5 * height, IMAGE_TO_BODY)


def to_gray(msg: Image) -> np.ndarray | None:
    data = np.frombuffer(msg.data, np.uint8)
    if msg.pixel_format_type == 1 and data.size == msg.width * msg.height:  # L_INT8
        return data.reshape(msg.height, msg.width).copy()
    if data.size == msg.width * msg.height * 3:  # RGB_INT8
        rgb = data.reshape(msg.height, msg.width, 3).astype(np.float32)
        return (rgb @ np.array([0.299, 0.587, 0.114], np.float32)).astype(np.uint8)
    return None


# Gazebo IMU through PX4 (conservative: the same densities as the Xsens runs).
MSCKF_IMU_NOISE = ImuNoiseDensity(gyro_rad_s_sqrt_hz=0.004, accel_m_s2_sqrt_hz=0.08,
                                  gyro_bias_walk_rad_s2_sqrt_hz=1e-4, accel_bias_walk_m_s3_sqrt_hz=4e-5)


def msckf_measurement(pipeline: CompanionVo, msckf: CompanionMsckf, tracker: KltTracker, t_frame: int,
                      gray: np.ndarray, last_t: int, record=None,
                      variance_scale: tuple[float, float] | None = None) -> BodyVelocityMeasurement | None:
    """MSCKF output in the same form as the VO measurement (body-FRD velocity,
    N/E velocity in the autopilot heading frame for the map anchor)."""
    state = pipeline.state_at_camera_time(t_frame)
    q = pipeline.quaternion_at_camera_time(t_frame)
    feats = tracker.track(gray)
    if state is None or q is None:
        msckf.count("NO_ATTITUDE_OR_RANGE")
        return None
    roll, pitch, yaw, height = state
    range_m = height / (math.cos(roll) * math.cos(pitch))
    if record:
        record.write(json.dumps({"frame": t_frame, "q": list(q), "roll": roll, "pitch": pitch, "yaw": yaw,
                                 "range": range_m, "features": {str(k): v for k, v in feats.items()}}) + "\n")
    out = msckf.process(t_frame, feats, attitude_q=q, roll=roll, pitch=pitch, range_m=range_m)
    if out is None:
        return None
    if variance_scale is not None:
        sx, sz = variance_scale
        out = replace(out, variance_frd=(sx * out.variance_frd[0], sx * out.variance_frd[1], sz * out.variance_frd[2]))
    r_nb = body_to_ned(roll, pitch, yaw)          # autopilot attitude, not the MSCKF heading
    v_ne = r_nb @ np.asarray(out.velocity_frd)
    cov_ne = r_nb @ np.diag(out.variance_frd) @ r_nb.T
    start = last_t if last_t and t_frame - last_t < 300_000 else t_frame - 66_000
    vo = VisualOdomResult(t_frame, 0, "MSCKF", True, 0.0, 0.0, float(v_ne[0]), float(v_ne[1]), 0, 0.0, 0.0,
                          float(cov_ne[0, 0]), float(cov_ne[0, 1]), float(cov_ne[1, 1]), start)
    return BodyVelocityMeasurement(t_frame, out.velocity_frd, out.variance_frd, vo, float(v_ne[2]),
                                   height, (roll, pitch, yaw))


def flat(c) -> tuple[float, float, float]:
    """(nn, ne, ee) of a 2x2 N/E covariance, NaN if absent."""
    if c is None:
        return (float("nan"),) * 3
    return (round(c[0][0], 5), round(c[0][1], 5), round(c[1][1], 5))


def fix_covariance(cal: dict, fix, height_m: float):
    """Per-fix covariance: the match-quality model if the calibration
    selects it (use_quality_model) and the fix carries a sub-patch
    covariance, otherwise the calibrated one at the frame height.

    The quality model is always fitted as a diagnostic; until series M it
    was used whenever present, so the R of those tests was the quality
    model, not covariance_ne_m2."""
    per_source = cal.get("per_source_covariance_ne_m2")
    if per_source and getattr(fix, "map_source", None) in per_source:
        return per_source[fix.map_source]
    q = cal.get("quality_model")
    if (q is not None and cal.get("use_quality_model") is True
            and getattr(fix, "subpatch_cov_ne_m2", None) is not None):
        return quality_covariance(q["floor_m2"], q["k2"], fix.subpatch_cov_ne_m2)
    return covariance_at_height(cal, height_m)


def stamp_us(header) -> int:
    return header.stamp.sec * 1_000_000 + header.stamp.nsec // 1000


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--world", required=True)
    ap.add_argument("--camera-topic", default="/ofnav/vo_camera")
    ap.add_argument("--hfov", type=float, default=1.2)
    ap.add_argument("--mavlink", default="udpin:0.0.0.0:14540")
    ap.add_argument("--camera-offset-frd", type=float, nargs=3, default=(0.0, 0.0, 0.1),
                    help="camera position in body FRD (x500_vo_down: 0.1 m below the centre)")
    ap.add_argument("--max-latency-ms", type=float, default=200.0)
    ap.add_argument("--no-vo-jump-gate", action="store_true",
                    help="send every VO velocity (default: reject jumps beyond the vehicle's acceleration, "
                         "src/companion_vo.py VelocityJumpGate)")
    ap.add_argument("--focal-error", type=float, default=0.0,
                    help="injected relative focal-length error (VO scale fault)")
    ap.add_argument("--map-mode", choices=("off", "observe", "fuse"), default="off",
                    help="observe: log orthophoto fixes only; fuse: send map-anchored position")
    ap.add_argument("--orthophoto", type=Path, help="map image (uint8 gray)")
    ap.add_argument("--surface", type=Path,
                    help="ncc: surface model on the map grid (.npy, metres up; swiss_scene_3d.py): the frame is "
                         "projected onto it instead of a flat ground")
    ap.add_argument("--orthophoto-fallback", type=Path,
                    help="ncc: second map tried when the first gives no fix (e.g. the full orthophoto behind "
                         "a ground-only map); each fix carries its source for the calibration")
    ap.add_argument("--map-georef", type=Path, help="JSON: origin_ne_m, matrix_ne_m_per_px")
    ap.add_argument("--map-calibration", type=Path,
                    help="JSON from calibrate_map_error: bias_ne_m, covariance_ne_m2 (fuse mode)")
    ap.add_argument("--map-rate-hz", type=float, default=2.0)
    ap.add_argument("--no-ev-pos-repeat-scale", dest="ev_pos_repeat_scale", action="store_false",
                    help="send the anchor position variance as is with every VO message (~15 Hz). By "
                         "default it is scaled by (map fix interval / message interval): one fix per "
                         "map interval (2 Hz), which EKF2 would otherwise fuse 7-8 times as independent. "
                         "No wind: EKF2 velocity error 0.35 -> 0.09 m/s, route max 1.6-1.7 -> 0.5-0.6 m "
                         "(zone-calibrated field), 1.2 -> 1.0 m (field A)")
    ap.add_argument("--acquisition-prior-sigma-m", type=float, default=3.0,
                    help="first map lock: autopilot (GNSS) position as prior with this sigma per axis")
    ap.add_argument("--max-fix-age-s", type=float, default=10.0,
                    help="send the anchor position only this long after the last fused map fix")
    ap.add_argument("--map-hold", type=float, nargs=2, metavar=("AGE_S", "MOVED_M"), default=[30.0, 3.0],
                    help="past --max-fix-age-s keep sending it up to AGE_S while the VO displacement "
                         "since the fix is <= MOVED_M (hover over a place without map fixes; swiss "
                         "start: 21.6 s without a fix, EKF2 lost the position 5 s after GNSS loss)")
    ap.add_argument("--no-map-hold", dest="map_hold", action="store_const", const=None,
                    help="no hold: the position stops --max-fix-age-s after the last fix")
    ap.add_argument("--max-map-step-error-m", type=float, default=1.0,
                    help="map displacement minus VO displacement: base engineering bound, "
                         "plus accumulated scale/heading/gap allowance (not flight validated)")
    ap.add_argument("--map-method", choices=("orb", "ncc"), default="orb",
                    help="ncc: ortho-rectified frame + correlation around a position prior "
                         "(orthophoto of another year/season)")
    ap.add_argument("--map-weak-match", type=float, nargs=2, metavar=("MARGIN", "PEAK"),
                    help="ncc: reject matches below both (OrthoNccLocalizer weak_match; strong-gust "
                         "calibration flights: 0.10 0.52)")
    ap.add_argument("--map-patch-at-boresight", action="store_true",
                    help="ncc: centre the map patch where the optical axis meets the ground (steady tilt, wind)")
    ap.add_argument("--map-max-tilt-rad", type=float, default=0.12,
                    help="ncc: largest roll/pitch for map matching (OrthoNccLocalizer max_tilt_rad, <= 0.2)")
    ap.add_argument("--no-vo-vertical-gate", action="store_true",
                    help="send the raw range-rate vertical velocity (src/companion_vo.py VerticalVelocityGate)")
    ap.add_argument("--vo-gap-bridge", type=Path, default=DEFAULT_GAP_BRIDGE,
                    help="JSON from tools/calibrate_vo_gap_bridge.py: carry the map anchor across VO gaps "
                         "with the autopilot velocity (src/companion_map.py VelocityBridge); fuse mode, "
                         "default: the strong-gust calibration (series GB)")
    ap.add_argument("--no-vo-gap-bridge", action="store_true",
                    help="VO gaps as unknown motion (max_gap_speed_m_s), the anchor may drop its lock")
    ap.add_argument("--vo-max-tilt-rad", type=float, default=0.35,
                    help="largest roll/pitch of a VO frame pair (planar model, ContinuousPlanarOdometry, <= 0.4). "
                         "0.35 since series N1 (strong gusts: 10/10 flights pass all checks against 8/10 at "
                         "0.25, which lost VO above 14 deg tilt and once EKF2 position)")
    ap.add_argument("--map-yaw", action="store_true",
                    help="fuse mode: also send the map-referenced heading (EKF2_EV_CTRL bit 3)")
    ap.add_argument("--map-control", type=Path,
                    help="JSON polled in flight: {\"shift_ne_m\": [dn, de], \"outage\": bool} injects a map "
                         "fault or a map outage (no matching)")
    ap.add_argument("--map-log", type=Path)
    ap.add_argument("--save-frames", type=Path, help="directory: every --save-every-th frame as PNG")
    ap.add_argument("--save-every", type=int, default=75)
    ap.add_argument("--estimator", choices=("vo", "msckf"), default="vo",
                    help="msckf: tightly coupled VIO on the autopilot IMU (HIGHRES_IMU), initialised "
                         "without GNSS from the rangefinder")
    ap.add_argument("--msckf-calibration", type=Path,
                    help="JSON from tools/calibrate_msckf_covariance.py: variance scales (separate flights)")
    ap.add_argument("--attitude-stamp", choices=("publish", "sample"), default="sample",
                    help="attitude for VO rotation compensation: ATTITUDE_QUATERNION (stamped at EKF2 "
                         "publication, 0-12 ms after the IMU sample) or the autopilot ODOMETRY (stamped at "
                         "the IMU sample)")
    ap.add_argument("--vo-surface-depth", action="store_true",
                    help="VO: each feature's depth on the --surface DSM (needs --surface, --map-georef) "
                         "instead of one rangefinder height for the frame (swiss tests: safe, no route gain)")
    ap.add_argument("--vo-window-pairs", type=int, default=1,
                    help="send the mean VO velocity over this many contiguous frame pairs (window midpoint "
                         "stamp); keep 1: 4 pairs made the swiss test flights swing at the route corners")
    ap.add_argument("--frame-time-shift-ms", type=float, default=0.0,
                    help="added to every camera frame stamp before use; keep 0: the apparent 105 ms VO lead "
                         "(tools/measure_frame_time_shift.py) applied here made the test flights diverge")
    ap.add_argument("--vo-calibration", type=Path,
                    help="JSON from tools/calibrate_vo_covariance.py: horizontal variance scale of the camera VO "
                         "(separate calibration flights); applied to what EKF2 and the map anchor get, the log "
                         "keeps the model variance")
    ap.add_argument("--msckf-record", type=Path,
                    help="JSON lines: IMU samples and per-frame MSCKF inputs (offline replay)")
    ap.add_argument("--log", type=Path)
    ap.add_argument("--status-s", type=float, default=5.0)
    args = ap.parse_args()
    if args.vo_surface_depth and not (args.surface and args.map_georef):
        ap.error("--vo-surface-depth needs --surface and --map-georef")
    if not 1 <= args.vo_window_pairs <= 16:
        # The map anchor keeps 16 intervals: a longer window's midpoint
        # would lose the map position.
        ap.error("--vo-window-pairs must be 1..16")
    if not 0.0 < args.vo_max_tilt_rad <= 0.4:
        ap.error("--vo-max-tilt-rad must be in (0, 0.4]")
    if args.map_mode != "off" and not (args.orthophoto and args.map_georef):
        ap.error("--map-mode needs --orthophoto and --map-georef")
    if args.no_vo_gap_bridge or args.map_mode != "fuse":
        args.vo_gap_bridge = None
    if args.map_mode == "fuse" and not args.map_calibration:
        ap.error("--map-mode fuse needs --map-calibration (no uncalibrated map covariance)")
    if not all(math.isfinite(x) and x > 0 for x in (args.max_fix_age_s, args.max_map_step_error_m)):
        ap.error("--max-fix-age-s and --max-map-step-error-m must be finite and positive")
    if args.map_hold and not (math.isfinite(args.map_hold[0]) and args.map_hold[0] > 0
                              and math.isfinite(args.map_hold[1]) and args.map_hold[1] >= 0):
        ap.error("--map-hold: AGE_S > 0 and MOVED_M >= 0, both finite")

    clock_us = [0]          # companion clock = simulation time
    frames: queue.Queue = queue.Queue(maxsize=8)
    node = Node()
    node.subscribe(Clock, f"/world/{args.world}/clock",
                   lambda m: clock_us.__setitem__(0, m.sim.sec * 1_000_000 + m.sim.nsec // 1000))

    # Frame stamp shift, default 0. The 105 ms the VO velocity appears to
    # lead the truth (tools/measure_frame_time_shift.py) is not a stamp
    # error: applying it made all three test flights diverge.
    frame_shift_us = int(round(args.frame_time_shift_ms * 1000))
    def on_image(msg: Image) -> None:
        gray = to_gray(msg)
        if gray is None:
            return
        try:
            frames.put_nowait((stamp_us(msg.header) + frame_shift_us, gray))
        except queue.Full:
            pass  # processing is behind: drop, the VO pair gate restarts the segment
    node.subscribe(Image, args.camera_topic, on_image)

    link = mavutil.mavlink_connection(args.mavlink, source_system=1, source_component=197)
    pipeline: CompanionVo | None = None
    pending: list[tuple[int, np.ndarray]] = []
    log = args.log.open("w") if args.log else None
    record = args.msckf_record.open("w", buffering=1) if args.msckf_record else None
    msckf_scale = None
    if args.msckf_calibration:
        mc = json.loads(args.msckf_calibration.read_text())
        msckf_scale = (float(mc["horizontal"]["scale"]), float(mc["vertical"]["scale"]))
    vo_scale = 1.0
    if args.vo_calibration:
        vo_scale = float(json.loads(args.vo_calibration.read_text())["horizontal"]["scale"])
    if log:
        log.write("t_us,vx,vy,vz,var_x,var_y,var_z,vn,ve,vd,inliers,jump_rejected\n")
    sent, last_ts, last_status, last_rates = 0, 0.0, 0.0, 0.0
    # Smoothed interval between sent messages (sim s), for --ev-pos-repeat-scale.
    msg_dt_s: float | None = None
    last_sent_us: int | None = None
    localizer: OrthophotoLocalizer | None = None
    anchor: MapAnchor | None = None
    map_cal: dict | None = None
    fallback = None
    source = "primary"
    map_shift = np.zeros(2)
    map_outage = False
    vo_outage = False   # injected VO dropout (map control "vo_outage"; calibration flights)
    bridge = VelocityBridge.from_json(json.loads(args.vo_gap_bridge.read_text())) if args.vo_gap_bridge else None
    msckf: CompanionMsckf | None = None
    tracker: KltTracker | None = None
    last_msckf_t = 0
    local_ne = np.array([np.nan, np.nan])
    # With map heading sent to EKF2: offsets relative to a gyro-integrated
    # heading, which the EV yaw fusion does not rotate (no feedback loop).
    gyro = GyroHeading() if args.map_yaw else None
    error_field: MapErrorField | None = None
    last_map_t, last_control, frame_no = 0, 0.0, 0
    map_counts: dict[str, int] = {}
    map_log = args.map_log.open("w") if args.map_log else None
    if map_log:
        map_log.write("t_us,localizer_status,north_m,east_m,inliers,heading_offset_rad,height_m,"
                      "injected_shift_n,injected_shift_e,anchor_result,nis,process_ms,"
                      "sub_nn,sub_ne,sub_ee,shape_nn,shape_ne,shape_ee,peak,margin,map_source\n")

    # MAVLink is read in its own thread: the MSCKF initialisation takes
    # 0.7-1.8 s, and a loop blocked that long overflowed the UDP buffer
    # (HIGHRES_IMU lost for seconds -> RESET_IMU_GAP -> a new initialisation).
    inbox: queue.Queue = queue.Queue()

    def reader() -> None:
        while True:
            m = link.recv_match(blocking=True, timeout=0.1)
            if m is None:
                continue
            arrival = clock_us[0]
            if m.get_type() == "TIMESYNC" and m.get_srcComponent() == 1 and m.tc1 == 0:
                # PX4 asks: answer at once with the companion clock.
                link.mav.timesync_send(arrival * 1000, m.ts1)
                continue
            inbox.put((m, arrival))
    threading.Thread(target=reader, daemon=True).start()

    while True:
        # Drain everything queued: a late TIMESYNC answer spoils both offsets.
        msgs = []
        try:
            msgs.append(inbox.get(timeout=0.01))
            while True:
                msgs.append(inbox.get_nowait())
        except queue.Empty:
            pass
        now = clock_us[0]
        for msg, arrival in msgs:
            if pipeline is None:
                break
            t = msg.get_type()
            if t == "TIMESYNC" and msg.get_srcComponent() == 1:
                # Answer to our request, at its arrival time.
                pipeline.clock.add(msg.ts1 // 1000, msg.tc1 // 1000, arrival)
            elif (t == "ODOMETRY" and args.attitude_stamp == "sample" and msg.get_srcComponent() == 1
                  and all(map(math.isfinite, msg.q))):
                # The autopilot's own odometry: stamped at the IMU sample
                # (vehicle_odometry.timestamp_sample).
                pipeline.attitude.add(int(msg.time_usec), tuple(float(v) for v in msg.q))
            elif t == "ATTITUDE_QUATERNION":
                if args.attitude_stamp == "publish":
                    pipeline.attitude.add(msg.time_boot_ms * 1000, (msg.q1, msg.q2, msg.q3, msg.q4))
                if gyro is not None:
                    gyro.add(msg.time_boot_ms * 1000, *quaternion_to_rpy(np.array([msg.q1, msg.q2, msg.q3, msg.q4])),
                             (msg.rollspeed, msg.pitchspeed, msg.yawspeed))
            elif t == "HIGHRES_IMU" and msckf is not None and pipeline.clock.valid:
                reading = ImuReading(int(msg.time_usec) - pipeline.clock.offset_us,
                                     (msg.xgyro, msg.ygyro, msg.zgyro), (msg.xacc, msg.yacc, msg.zacc))
                msckf.add_imu(reading)
                if record:
                    record.write(json.dumps({"imu": reading.timestamp_us, "g": list(reading.gyro_body_rad_s),
                                             "a": list(reading.specific_force_body_m_s2)}) + "\n")
            elif t == "LOCAL_POSITION_NED":
                local_ne = np.array([msg.x, msg.y])
                if bridge is not None and pipeline.clock.valid:
                    bridge.add(msg.time_boot_ms * 1000 - pipeline.clock.offset_us, msg.vx, msg.vy)
                # Before the first lock nothing of ours is fused yet: the
                # autopilot position is GNSS-aided and independent of the map.
                if anchor is not None and not anchor.counts.get("LOCK_ACQUIRED"):
                    anchor.set_acquisition_prior(local_ne, np.eye(2) * args.acquisition_prior_sigma_m ** 2)
                if record and pipeline.clock.valid:
                    record.write(json.dumps({"lpos": msg.time_boot_ms * 1000 - pipeline.clock.offset_us,
                                             "v": [msg.vx, msg.vy, msg.vz]}) + "\n")
            elif t == "DISTANCE_SENSOR" and msg.orientation == 25 and msg.current_distance > 0:
                pipeline.distance.add(msg.time_boot_ms * 1000, msg.current_distance * 0.01)
        if pipeline is None and link.target_system and now > 0:
            # First contact: size the camera model from the first frame.
            try:
                t0, first = frames.get(timeout=1.0)
            except queue.Empty:
                continue
            h, w = first.shape
            cal = calibration(w, h, args.hfov, args.focal_error)
            vo_surface = None
            if args.vo_surface_depth:
                vo_surface = SurfaceDepth.from_georef(np.load(args.surface),
                                                      json.loads(args.map_georef.read_text()))
            vo = ContinuousPlanarOdometry(
                cal, camera_mount=CameraMountCalibration(BODY_FROM_CAMERA),
                camera_offset_body_m=tuple(args.camera_offset_frd),
                velocity_error_assumptions=VelocityErrorAssumptions(
                    fitted_displacement_sigma_px=0.5, height_sigma_m=0.05, interval_sigma_s=0.002,
                    yaw_sigma_rad=0.02, focal_relative_sigma=0.01, velocity_model_floor_m_s=0.05),
                max_tilt_rad=args.vo_max_tilt_rad, max_pair_dt_s=0.25, surface=vo_surface)
            pipeline = CompanionVo(vo, window_pairs=args.vo_window_pairs,
                                   vertical_gate=None if args.no_vo_vertical_gate else VerticalVelocityGate())
            # The gate's acceleration bound (3.7 m/s^2) is derived from the 0.25 rad
            # VO tilt limit; it scales with tan(limit) when the limit is changed.
            jump_accel = 3.7 * math.tan(args.vo_max_tilt_rad) / math.tan(0.25)
            jump_gate = (None if args.no_vo_jump_gate or args.estimator == "msckf"
                         else VelocityJumpGate(max_accel_m_s2=jump_accel))
            if args.estimator == "msckf":
                msckf = CompanionMsckf(
                    PinholeCamera(cal.focal_x_px, cal.focal_y_px, cal.center_x_px, cal.center_y_px,
                                  BODY_FROM_CAMERA, tuple(args.camera_offset_frd), pixel_sigma=1.0),
                    MSCKF_IMU_NOISE)
                tracker = KltTracker(max_features=100)
            if args.map_mode != "off":
                import cv2
                # Radiometric normalisation of map and frame: the rendered
                # frames are dark and low-contrast; ORB matches none raw.
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
                georef = json.loads(args.map_georef.read_text())
                geo = MapGeoReference(tuple(georef["origin_ne_m"]),
                                      tuple(tuple(r) for r in georef["matrix_ne_m_per_px"]))
                map_gray = cv2.imread(str(args.orthophoto), cv2.IMREAD_GRAYSCALE)
                if args.map_method == "ncc":
                    surface = np.load(args.surface) if args.surface else None
                    localizer = OrthoNccLocalizer(map_gray, geo, cal, CameraMountCalibration(BODY_FROM_CAMERA),
                                                  camera_offset_body_m=tuple(args.camera_offset_frd),
                                                  surface_m=surface, patch_at_boresight=args.map_patch_at_boresight,
                                                  max_tilt_rad=args.map_max_tilt_rad,
                                                  weak_match=tuple(args.map_weak_match) if args.map_weak_match else None)
                    if args.orthophoto_fallback:
                        fallback = OrthoNccLocalizer(cv2.imread(str(args.orthophoto_fallback), cv2.IMREAD_GRAYSCALE),
                                                     geo, cal, CameraMountCalibration(BODY_FROM_CAMERA),
                                                     camera_offset_body_m=tuple(args.camera_offset_frd),
                                                     surface_m=surface,
                                                     patch_at_boresight=args.map_patch_at_boresight,
                                                     max_tilt_rad=args.map_max_tilt_rad,
                                                  weak_match=tuple(args.map_weak_match) if args.map_weak_match else None)
                else:
                    tile = OrthophotoTile("orthophoto", clahe.apply(map_gray), geo)
                    localizer = OrthophotoLocalizer([tile], cal, CameraMountCalibration(BODY_FROM_CAMERA),
                                                    camera_offset_body_m=tuple(args.camera_offset_frd))
            if args.map_mode == "fuse":
                c = map_cal = json.loads(args.map_calibration.read_text())
                if c.get("use_error_field") and c.get("error_field"):
                    # Offset and variance from the kriged field at each fix;
                    # the anchor then has a zero bias.
                    error_field = MapErrorField.from_json(c["error_field"])
                anchor = MapAnchor(bias_ne_m=(0.0, 0.0) if (c.get("per_source_bias_ne_m") or error_field is not None)
                                   else tuple(c["bias_ne_m"]),
                                   covariance_ne_m2=tuple(tuple(r) for r in c["covariance_ne_m2"]),
                                   max_map_step_error_m=args.max_map_step_error_m)
            pending.append((t0, first))
            print(json.dumps({"camera": [w, h], "focal_px": round(cal.focal_x_px, 1)}), flush=True)
        if pipeline is None:
            continue
        wall = time.time()
        if wall - last_ts > 0.2 and now > 0:
            link.mav.timesync_send(0, now * 1000)
            last_ts = wall
        if wall - last_rates > 5.0:
            streams = [(mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE_QUATERNION, 100),
                       (mavutil.mavlink.MAVLINK_MSG_ID_DISTANCE_SENSOR, 50)]
            if bridge is not None:
                streams.append((mavutil.mavlink.MAVLINK_MSG_ID_LOCAL_POSITION_NED, 50))
            if args.attitude_stamp == "sample":
                streams.append((mavutil.mavlink.MAVLINK_MSG_ID_ODOMETRY, 100))
            if msckf is not None:
                streams.append((mavutil.mavlink.MAVLINK_MSG_ID_HIGHRES_IMU, 200))
            for msg_id, hz in streams:
                link.mav.command_long_send(link.target_system, 1,
                                           mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                                           msg_id, 1e6 / hz, 0, 0, 0, 0, 0)
            last_rates = wall
        while not frames.empty():
            pending.append(frames.get_nowait())
        if args.map_control and wall - last_control > 0.5:
            last_control = wall
            try:
                ctl = json.loads(args.map_control.read_text())
                map_shift = np.asarray(ctl.get("shift_ne_m", (0, 0)), float)
                map_outage = bool(ctl.get("outage", False))
                vo_outage = bool(ctl.get("vo_outage", False))
            except (OSError, ValueError):
                pass
        # Process frames in order once autopilot samples exist after them.
        while pending and pipeline.ready_for(pending[0][0]) and (msckf is None or msckf.ready_for(pending[0][0])):
            t_frame, gray = pending.pop(0)
            frame_no += 1
            if args.save_frames and frame_no % args.save_every == 0:
                import cv2
                args.save_frames.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(args.save_frames / f"{t_frame}.png"), gray)
            if vo_outage:
                out = None
                pipeline.count("VO_OUTAGE")
            elif msckf is not None:
                out = msckf_measurement(pipeline, msckf, tracker, t_frame, gray, last_msckf_t, record, msckf_scale)
                if out is not None:
                    last_msckf_t = t_frame
            else:
                # DSM lookup position: the anchor once locked, else the autopilot's.
                st = anchor.status() if anchor is not None else None
                pos = st.position_ne_m if st is not None and st.locked else local_ne
                out = pipeline.process(t_frame, gray, position_ne_m=(float(pos[0]), float(pos[1])))
            vo_rejected = False
            # The gate checks what is sent: the window mean when windowing.
            v_gate = out.velocity_ne_sent if out is not None and out.pairs_in_window > 1 else \
                ((out.vo.vn_m_s, out.vo.ve_m_s) if out is not None else None)
            if out is not None and jump_gate is not None and not jump_gate.check(out.time_us, v_gate):
                vo_rejected = True
                pipeline.count("VO_JUMP_REJECTED")
            if out is not None and anchor is not None:
                r = out.vo
                v_prop = (r.vn_m_s, r.ve_m_s)
                k = vo_scale if msckf is None else 1.0
                c_prop = ((k * r.velocity_cov_nn, k * r.velocity_cov_ne), (k * r.velocity_cov_ne, k * r.velocity_cov_ee))
                if vo_rejected:
                    # Across a rejected interval: the last accepted velocity,
                    # its variance grown by the possible acceleration.
                    v_prop, var_h = jump_gate.held_velocity(out.time_us)
                    c_prop = ((var_h, 0.0), (0.0, var_h))
                ref = 0.0
                if gyro is not None:
                    d = gyro.at(out.time_us + pipeline.clock.offset_us) - out.rpy_mid[2]
                    ref = math.atan2(math.sin(d), math.cos(d)) if math.isfinite(d) else float("nan")
                gap_d = gap_sigma = None
                if bridge is not None and anchor.time_us and r.interval_start_us - anchor.time_us > 1000:
                    b = bridge.displacement(anchor.time_us, r.interval_start_us)
                    if b is not None:
                        gap_d, gap_sigma = (float(b[0][0]), float(b[0][1])), b[1]
                    else:
                        pipeline.count("VO_GAP_NOT_BRIDGED")
                anchor.propagate(r.interval_start_us, r.timestamp_us, v_prop, c_prop,
                                 reference_minus_autopilot_rad=ref, gap_displacement_ne=gap_d,
                                 gap_sigma_m=gap_sigma)
            if localizer is not None and map_outage and t_frame - last_map_t >= 1e6 / args.map_rate_hz:
                # Injected outage (map control): no map matching at all.
                last_map_t = t_frame
                pipeline.count("MAP_OUTAGE")
            elif localizer is not None and t_frame - last_map_t >= 1e6 / args.map_rate_hz:
                state = pipeline.state_at_camera_time(t_frame)
                if state is not None:
                    last_map_t = t_frame
                    t_start = time.perf_counter()
                    if args.map_method == "ncc":
                        # Position prior: the anchor once locked, otherwise the
                        # autopilot's own local position (GNSS-aided before loss).
                        st = anchor.status() if anchor is not None else None
                        if st is not None and st.locked:
                            prior = np.asarray(st.position_ne_m)
                            radius = max(10.0, 3 * math.sqrt(max(st.covariance_ne_m2[0][0],
                                                                 st.covariance_ne_m2[1][1])) + 5.0)
                        else:
                            prior, radius = local_ne, 20.0
                        fix = localizer.localize(gray, prior_ne_m=prior, radius_m=radius, height_agl_m=state[3],
                                                 roll_rad=state[0], pitch_rad=state[1], yaw_rad=state[2])
                        source = "primary"
                        if not fix.accepted and fallback is not None:
                            fix2 = fallback.localize(gray, prior_ne_m=prior, radius_m=radius, height_agl_m=state[3],
                                                     roll_rad=state[0], pitch_rad=state[1], yaw_rad=state[2])
                            if fix2.accepted:
                                fix, source = fix2, "fallback"
                    else:
                        fix = localizer.localize(clahe.apply(gray), height_agl_m=state[3], roll_rad=state[0],
                                                 pitch_rad=state[1], yaw_rad=state[2])
                    ms = (time.perf_counter() - t_start) * 1e3
                    n, e = fix.north_m + map_shift[0], fix.east_m + map_shift[1]
                    # Per-source bias (calibrated separately); the anchor then has a zero bias.
                    if map_cal and map_cal.get("per_source_bias_ne_m") and source in map_cal["per_source_bias_ne_m"]:
                        bs = map_cal["per_source_bias_ne_m"][source]
                        n, e = n - bs[0], e - bs[1]
                    field_cov = None
                    if error_field is not None and fix.accepted and math.isfinite(n) and math.isfinite(e):
                        m_f, _ = error_field.predict([[n, e]])
                        var_f = float(error_field.fix_variance([[n, e]])[0])
                        n, e = n - float(m_f[0][0]), e - float(m_f[0][1])
                        field_cov = ((var_f, 0.0), (0.0, var_f))
                    result, nis = "", float("nan")
                    if anchor is not None:
                        # Position is known at the end of the last VO interval only.
                        if out is not None and out.vo.timestamp_us == t_frame:
                            ref_yaw = state[2] if gyro is None else gyro.at(t_frame + pipeline.clock.offset_us)
                            psi = math.atan2(math.sin(fix.yaw_rad - ref_yaw), math.cos(fix.yaw_rad - ref_yaw))
                            result = anchor.fix(t_frame, fix.accepted, n, e, psi,
                                                covariance_ne_m2=field_cov if field_cov is not None else fix_covariance(
                                                    map_cal, replace(fix, map_source=source), state[3])
                                                if fix.accepted else None)
                            nis = anchor.last_nis if result in ("FIX_FUSED", "FIX_REJECTED_NIS", "FIX_REJECTED_STEP") else nis
                        else:
                            result = "NO_VO_AT_FRAME"
                    map_counts[fix.status] = map_counts.get(fix.status, 0) + 1
                    if map_log:
                        map_log.write(",".join(str(v) for v in (
                            t_frame, fix.status, round(n, 4), round(e, 4), fix.inlier_points,
                            round(fix.yaw_rad - state[2], 4), round(state[3], 3), map_shift[0], map_shift[1],
                            result, round(nis, 3), round(ms, 1), *flat(fix.subpatch_cov_ne_m2), *flat(fix.shape_ne_m2),
                            round(float(getattr(localizer, "last_peak", float("nan"))), 4),
                            round(float(getattr(localizer, "last_margin", float("nan"))), 4), source)) + "\n")
            if out is None:
                continue
            # A late measurement is worse than none: EKF2 fuses it at its stamp
            # only if PX4's timesync has converged, otherwise at arrival.
            if clock_us[0] - out.time_us > args.max_latency_ms * 1000:
                pipeline.count("TOO_LATE")
                continue
            nan = float("nan")
            cov = [0.0] * 21
            cov[0], cov[6], cov[11] = out.variance_frd
            if msckf is None:
                cov[0] *= vo_scale
                cov[6] *= vo_scale
            cov[15] = cov[18] = cov[20] = nan
            velocity = list(out.velocity_frd)
            if vo_rejected:
                # No velocity measurement; the map position (if any) still goes.
                velocity = [nan, nan, nan]
                cov = [nan] * 21
            # Position only with a map lock: EKF2 stops position fusion on any
            # message without it, so it is sent with every velocity message.
            pos, pose_cov, quat = [nan, nan, nan], [nan] * 21, [nan, nan, nan, nan]
            # Only a map-backed position: without a fused fix for max_fix_age_s
            # the anchor is VO integrated with the autopilot heading, and EKF2
            # fusing it next to the VO velocity rotated the heading by 17 deg
            # after GNSS loss (swiss scene). Velocity only then.
            hold = dict(hold_age_s=args.map_hold[0], hold_moved_m=args.map_hold[1]) if args.map_hold else {}
            measurement = anchor.measurement_at(out.time_us, max_fix_age_s=args.max_fix_age_s, **hold) \
                if anchor is not None else None
            if measurement is not None and math.isfinite(out.height_mid_m):
                pos = [*measurement.position_ne_m,
                       -out.height_mid_m]  # map datum = ground plane; EV height is not fused
                pose_cov = list(measurement.mavlink_pose_covariance(0.05 ** 2))
                if args.ev_pos_repeat_scale and msg_dt_s is not None:
                    k = min(30.0, max(1.0, (1.0 / args.map_rate_hz) / msg_dt_s))
                    for i in (0, 1, 6):           # N-N, N-E, E-E of the pose covariance
                        pose_cov[i] *= k
                r, p_, y = out.rpy_mid
                if gyro is not None:
                    # Not the autopilot heading: EKF2 turns it towards what
                    # we send, and the offset would be added a second time.
                    y = gyro.at(out.time_us + pipeline.clock.offset_us)
                if args.map_yaw and anchor.heading_known and all(map(math.isfinite, (r, p_, y))):
                    # Heading = reference heading + offset measured by the map.
                    # EKF2 takes only the yaw of this quaternion (fuseYaw);
                    # roll and pitch are the autopilot's own.
                    y += anchor.heading_offset_rad
                    cr, sr, cp, sp, cy, sy = (math.cos(r / 2), math.sin(r / 2), math.cos(p_ / 2),
                                              math.sin(p_ / 2), math.cos(y / 2), math.sin(y / 2))
                    quat = [cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                            cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy]
                    pose_cov[15] = pose_cov[18] = 0.1 ** 2
                    pose_cov[20] = anchor.heading_sigma_rad() ** 2
            link.mav.odometry_send(
                out.time_us, mavutil.mavlink.MAV_FRAME_LOCAL_NED, mavutil.mavlink.MAV_FRAME_BODY_FRD,
                *pos, quat, *velocity, nan, nan, nan,
                pose_cov, cov, 0, mavutil.mavlink.MAV_ESTIMATOR_TYPE_VIO, 100)
            sent += 1
            if last_sent_us is not None and 0 < out.time_us - last_sent_us < 1_000_000:
                dt = (out.time_us - last_sent_us) * 1e-6
                msg_dt_s = dt if msg_dt_s is None else 0.9 * msg_dt_s + 0.1 * dt
            last_sent_us = out.time_us
            if log:
                r = out.vo
                # Variances with 6 significant digits: calibrated MSCKF
                # variances are ~1e-5 m^2/s^2 and rounded to 0 at 5 decimals.
                log.write(",".join([str(out.time_us + pipeline.clock.offset_us)]
                                   + [str(round(v, 5)) for v in out.velocity_frd]
                                   + [f"{v:.6g}" for v in out.variance_frd]
                                   + [str(round(v, 5)) for v in (r.vn_m_s, r.ve_m_s, out.vd_m_s)])
                          + f",{r.inlier_count},{int(vo_rejected)}\n")
        # A frame waiting far too long has no autopilot data around it.
        while pending and now - pending[0][0] > 1_000_000:
            pending.pop(0)
            pipeline.count("STALE_FRAME")
        if time.time() - last_status > args.status_s:
            last_status = time.time()
            status = {"sim_s": round(now * 1e-6, 1), "sent": sent,
                      "time_offset_us": pipeline.clock.offset_us if pipeline.clock.valid else None,
                      "counts": pipeline.counts}
            if msckf is not None:
                status["msckf"] = {"counts": msckf.counts, "last_init": msckf.last_init,
                                   "last_error": msckf.last_error}
            if localizer is not None:
                status["map_localizer"] = map_counts
            if anchor is not None:
                st = anchor.status()
                status["map_anchor"] = {"locked": st.locked, "counts": st.counts,
                                        "step_error_m": anchor.last_step_error_m,
                                        "ci_weight_prior": anchor.last_ci_weight_prior,
                                        "heading_offset_deg": round(math.degrees(st.heading_offset_rad), 2),
                                        "position_ne_m": st.position_ne_m,
                                        "sigma_ne_m": [math.sqrt(st.covariance_ne_m2[i][i]) for i in (0, 1)]
                                        if st.locked else None}
            print(json.dumps(status), flush=True)
            if log:
                log.flush()
            if map_log:
                map_log.flush()


if __name__ == "__main__":
    main()
