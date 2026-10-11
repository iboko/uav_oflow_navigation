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
import time
from pathlib import Path

import numpy as np
from gz.msgs10.clock_pb2 import Clock
from gz.msgs10.image_pb2 import Image
from gz.transport13 import Node
from pymavlink import mavutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.camera_rotation_compensation import CameraMountCalibration  # noqa: E402
from src.companion_vo import CompanionVo  # noqa: E402
from src.continuous_visual_odometry import ContinuousPlanarOdometry  # noqa: E402
from src.ground_visual_motion import CameraCalibration  # noqa: E402
from src.velocity_uncertainty import VelocityErrorAssumptions  # noqa: E402

# Camera pitched +90 deg about body y (gz model): image right = body right,
# image down = body backward. Optical axis = body down.
IMAGE_TO_BODY = ((0.0, -1.0), (1.0, 0.0))
BODY_FROM_CAMERA = ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))


def calibration(width: int, height: int, hfov_rad: float) -> CameraCalibration:
    f = 0.5 * width / math.tan(0.5 * hfov_rad)
    return CameraCalibration(f, f, 0.5 * width, 0.5 * height, IMAGE_TO_BODY)


def to_gray(msg: Image) -> np.ndarray | None:
    data = np.frombuffer(msg.data, np.uint8)
    if msg.pixel_format_type == 1 and data.size == msg.width * msg.height:  # L_INT8
        return data.reshape(msg.height, msg.width).copy()
    if data.size == msg.width * msg.height * 3:  # RGB_INT8
        rgb = data.reshape(msg.height, msg.width, 3).astype(np.float32)
        return (rgb @ np.array([0.299, 0.587, 0.114], np.float32)).astype(np.uint8)
    return None


def stamp_us(header) -> int:
    return header.stamp.sec * 1_000_000 + header.stamp.nsec // 1000


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--world", required=True)
    ap.add_argument("--camera-topic", default="/ofnav/vo_camera")
    ap.add_argument("--hfov", type=float, default=1.2)
    ap.add_argument("--mavlink", default="udpin:0.0.0.0:14540")
    ap.add_argument("--camera-offset-frd", type=float, nargs=3, default=(0.0, 0.0, -0.1))
    ap.add_argument("--max-latency-ms", type=float, default=200.0)
    ap.add_argument("--log", type=Path)
    ap.add_argument("--status-s", type=float, default=5.0)
    args = ap.parse_args()

    clock_us = [0]          # companion clock = simulation time
    frames: queue.Queue = queue.Queue(maxsize=8)
    node = Node()
    node.subscribe(Clock, f"/world/{args.world}/clock",
                   lambda m: clock_us.__setitem__(0, m.sim.sec * 1_000_000 + m.sim.nsec // 1000))

    def on_image(msg: Image) -> None:
        gray = to_gray(msg)
        if gray is None:
            return
        try:
            frames.put_nowait((stamp_us(msg.header), gray))
        except queue.Full:
            pass  # processing is behind: drop, the VO pair gate restarts the segment
    node.subscribe(Image, args.camera_topic, on_image)

    link = mavutil.mavlink_connection(args.mavlink, source_system=1, source_component=197)
    pipeline: CompanionVo | None = None
    pending: list[tuple[int, np.ndarray]] = []
    log = args.log.open("w") if args.log else None
    if log:
        log.write("t_us,vx,vy,vz,var_x,var_y,var_z,vn,ve,vd,inliers\n")
    sent, last_ts, last_status, last_rates = 0, 0.0, 0.0, 0.0

    while True:
        # Drain everything queued: a late TIMESYNC answer spoils both offsets.
        msgs = [link.recv_match(blocking=True, timeout=0.01)]
        while msgs[-1] is not None:
            msgs.append(link.recv_match(blocking=False))
        now = clock_us[0]
        for msg in msgs[:-1]:
            if pipeline is None:
                break
            t = msg.get_type()
            if t == "TIMESYNC" and msg.get_srcComponent() == 1:
                if msg.tc1 == 0:      # PX4 asks: answer with the companion clock
                    link.mav.timesync_send(clock_us[0] * 1000, msg.ts1)
                else:                 # answer to our request
                    pipeline.clock.add(msg.ts1 // 1000, msg.tc1 // 1000, clock_us[0])
            elif t == "ATTITUDE_QUATERNION":
                pipeline.attitude.add(msg.time_boot_ms * 1000, (msg.q1, msg.q2, msg.q3, msg.q4))
            elif t == "DISTANCE_SENSOR" and msg.orientation == 25 and msg.current_distance > 0:
                pipeline.distance.add(msg.time_boot_ms * 1000, msg.current_distance * 0.01)
        if pipeline is None and link.target_system and now > 0:
            # First contact: size the camera model from the first frame.
            try:
                t0, first = frames.get(timeout=1.0)
            except queue.Empty:
                continue
            h, w = first.shape
            cal = calibration(w, h, args.hfov)
            vo = ContinuousPlanarOdometry(
                cal, camera_mount=CameraMountCalibration(BODY_FROM_CAMERA),
                camera_offset_body_m=tuple(args.camera_offset_frd),
                velocity_error_assumptions=VelocityErrorAssumptions(
                    fitted_displacement_sigma_px=0.5, height_sigma_m=0.05, interval_sigma_s=0.002,
                    yaw_sigma_rad=0.02, focal_relative_sigma=0.01, velocity_model_floor_m_s=0.05),
                max_tilt_rad=0.25, max_pair_dt_s=0.25)
            pipeline = CompanionVo(vo)
            pending.append((t0, first))
            print(json.dumps({"camera": [w, h], "focal_px": round(cal.focal_x_px, 1)}), flush=True)
        if pipeline is None:
            continue
        wall = time.time()
        if wall - last_ts > 0.2 and now > 0:
            link.mav.timesync_send(0, now * 1000)
            last_ts = wall
        if wall - last_rates > 5.0:
            for msg_id, hz in ((mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE_QUATERNION, 100),
                               (mavutil.mavlink.MAVLINK_MSG_ID_DISTANCE_SENSOR, 50)):
                link.mav.command_long_send(link.target_system, 1,
                                           mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                                           msg_id, 1e6 / hz, 0, 0, 0, 0, 0)
            last_rates = wall
        while not frames.empty():
            pending.append(frames.get_nowait())
        # Process frames in order once autopilot samples exist after them.
        while pending and pipeline.ready_for(pending[0][0]):
            t_frame, gray = pending.pop(0)
            out = pipeline.process(t_frame, gray)
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
            cov[15] = cov[18] = cov[20] = nan
            link.mav.odometry_send(
                out.time_us, mavutil.mavlink.MAV_FRAME_LOCAL_FRD, mavutil.mavlink.MAV_FRAME_BODY_FRD,
                nan, nan, nan, [nan, nan, nan, nan], *out.velocity_frd, nan, nan, nan,
                [nan] * 21, cov, 0, mavutil.mavlink.MAV_ESTIMATOR_TYPE_VIO, 100)
            sent += 1
            if log:
                r = out.vo
                log.write(",".join(str(round(v, 5)) for v in (
                    out.time_us + pipeline.clock.offset_us, *out.velocity_frd, *out.variance_frd,
                    r.vn_m_s, r.ve_m_s, out.vd_m_s)) + f",{r.inlier_count}\n")
        # A frame waiting far too long has no autopilot data around it.
        while pending and now - pending[0][0] > 1_000_000:
            pending.pop(0)
            pipeline.count("STALE_FRAME")
        if time.time() - last_status > args.status_s:
            last_status = time.time()
            print(json.dumps({"sim_s": round(now * 1e-6, 1), "sent": sent,
                              "time_offset_us": pipeline.clock.offset_us if pipeline.clock.valid else None,
                              "counts": pipeline.counts}), flush=True)
            if log:
                log.flush()


if __name__ == "__main__":
    main()
