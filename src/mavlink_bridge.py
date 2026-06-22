from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from pymavlink import mavutil
except ImportError as exc:
    raise SystemExit("Установите pymavlink: pip install pymavlink") from exc

try:
    from config import NavigationConfig
    from oflow_math import optical_flow_to_velocity, altitude_from_range
    from safety_monitor import OpticalFlowSafetyMonitor
except ImportError:
    from .config import NavigationConfig
    from .oflow_math import optical_flow_to_velocity, altitude_from_range
    from .safety_monitor import OpticalFlowSafetyMonitor


class MavlinkOpticalFlowBridge:
    """Диагностический MAVLink-мост для режима ОП.

    Назначение:
    - читать OPTICAL_FLOW_RAD, DISTANCE_SENSOR, ATTITUDE/HIGHRES_IMU;
    - пересчитывать ОП в горизонтальную скорость;
    - писать диагностический CSV;
    - при необходимости отправлять VISION_SPEED_ESTIMATE.

    Для штатного PX4/ArduPilot чаще правильнее подавать в автопилот
    исходные сообщения датчика ОП и дальномера, а не внешний EKF.
    Включайте отправку внешней скорости только после настройки EKF
    под External Vision / Vision Speed на конкретной прошивке.
    """

    def __init__(
        self,
        connection: str,
        baud: int,
        out_csv: str | Path,
        cfg: NavigationConfig,
        send_vision_speed: bool = False,
    ):
        self.master = mavutil.mavlink_connection(connection, baud=baud)
        self.out_csv = Path(out_csv)
        self.cfg = cfg
        self.send_vision_speed = send_vision_speed
        self.monitor = OpticalFlowSafetyMonitor(cfg)

        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.range_m = math.nan
        self.last_time_s = 0.0

    def wait_heartbeat(self) -> None:
        print("Ожидание heartbeat...")
        self.master.wait_heartbeat()
        print(
            f"Heartbeat: system={self.master.target_system}, component={self.master.target_component}"
        )

    def run(self) -> None:
        self.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with self.out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "time_s",
                    "height_m",
                    "range_m",
                    "roll_rad",
                    "pitch_rad",
                    "yaw_rad",
                    "flow_quality",
                    "vx_body_mps",
                    "vy_body_mps",
                    "vx_nav_mps",
                    "vy_nav_mps",
                    "valid",
                    "reason",
                    "safety_mode",
                    "warnings",
                ],
            )
            writer.writeheader()

            while True:
                msg = self.master.recv_match(blocking=True, timeout=1.0)
                now_s = time.time()
                if msg is None:
                    continue

                mtype = msg.get_type()

                if mtype == "ATTITUDE":
                    self.roll = float(msg.roll)
                    self.pitch = float(msg.pitch)
                    self.yaw = float(msg.yaw)
                    continue

                if mtype == "DISTANCE_SENSOR":
                    # MAVLink DISTANCE_SENSOR current_distance обычно в сантиметрах.
                    self.range_m = float(msg.current_distance) / 100.0
                    continue

                if mtype != "OPTICAL_FLOW_RAD":
                    continue

                dt_s = float(msg.integration_time_us) * 1e-6
                quality = float(msg.quality)
                height_m = altitude_from_range(self.range_m, self.roll, self.pitch)

                status = self.monitor.update(
                    t_s=now_s,
                    range_m=self.range_m,
                    height_m=height_m,
                    flow_quality=quality,
                    imu_valid=True,
                    flow_received=True,
                    range_received=np.isfinite(self.range_m),
                )

                res = optical_flow_to_velocity(
                    integrated_x_rad=float(msg.integrated_x),
                    integrated_y_rad=float(msg.integrated_y),
                    integrated_xgyro_rad=float(msg.integrated_xgyro),
                    integrated_ygyro_rad=float(msg.integrated_ygyro),
                    dt_s=dt_s,
                    range_m=self.range_m,
                    roll_rad=self.roll,
                    pitch_rad=self.pitch,
                    yaw_rad=self.yaw,
                    quality=quality,
                    cfg=self.cfg,
                )

                allow = bool(res.valid and status.allow_optical_flow)

                if allow and self.send_vision_speed:
                    usec = int(time.time() * 1_000_000)
                    # VISION_SPEED_ESTIMATE: usec, x, y, z, covariance[9]
                    cov = [0.05, 0, 0, 0, 0.05, 0, 0, 0, 0.5]
                    self.master.mav.vision_speed_estimate_send(
                        usec,
                        float(res.vx_nav_mps),
                        float(res.vy_nav_mps),
                        0.0,
                        cov,
                    )

                writer.writerow(
                    {
                        "time_s": now_s,
                        "height_m": height_m,
                        "range_m": self.range_m,
                        "roll_rad": self.roll,
                        "pitch_rad": self.pitch,
                        "yaw_rad": self.yaw,
                        "flow_quality": quality,
                        "vx_body_mps": res.vx_body_mps,
                        "vy_body_mps": res.vy_body_mps,
                        "vx_nav_mps": res.vx_nav_mps,
                        "vy_nav_mps": res.vy_nav_mps,
                        "valid": int(allow),
                        "reason": res.reason,
                        "safety_mode": status.mode,
                        "warnings": ",".join(status.warnings),
                    }
                )
                f.flush()

                print(
                    f"ОП valid={allow} mode={status.mode} "
                    f"Vnav=({res.vx_nav_mps:.3f}, {res.vy_nav_mps:.3f}) "
                    f"h={height_m:.2f} q={quality:.0f} reason={res.reason}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="MAVLink-мост диагностики оптического потока")
    parser.add_argument("--connection", default="udp:127.0.0.1:14540", help="MAVLink connection string")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--out", default="outputs/mavlink_oflow_diag.csv")
    parser.add_argument("--send-vision-speed", action="store_true", help="Отправлять VISION_SPEED_ESTIMATE")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = NavigationConfig.from_yaml(args.config) if args.config else NavigationConfig()
    bridge = MavlinkOpticalFlowBridge(
        connection=args.connection,
        baud=args.baud,
        out_csv=args.out,
        cfg=cfg,
        send_vision_speed=args.send_vision_speed,
    )
    bridge.wait_heartbeat()
    bridge.run()


if __name__ == "__main__":
    main()
