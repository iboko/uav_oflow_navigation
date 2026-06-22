from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from pymavlink import mavutil
except ImportError as exc:
    raise SystemExit("Установите pymavlink: pip install pymavlink") from exc


def send_distance_sensor(master, t_s: float, range_m: float) -> None:
    if not np.isfinite(range_m):
        return
    time_boot_ms = int(t_s * 1000)
    min_cm = 10
    max_cm = 500
    current_cm = int(np.clip(range_m * 100.0, min_cm, max_cm))
    sensor_type_laser = 0
    sensor_id = 1
    # MAV_SENSOR_ROTATION_PITCH_270 обычно соответствует направлению вниз для дальномера.
    # В разных версиях pymavlink можно использовать числовое значение.
    orientation_down = 25
    covariance = 3
    master.mav.distance_sensor_send(
        time_boot_ms,
        min_cm,
        max_cm,
        current_cm,
        sensor_type_laser,
        sensor_id,
        orientation_down,
        covariance,
    )


def send_optical_flow_rad(master, r) -> None:
    if not np.isfinite(r["of_integrated_x_rad"]):
        return

    time_usec = int(float(r["t_s"]) * 1_000_000)
    sensor_id = 1
    integration_time_us = int(float(r["of_dt_s"]) * 1_000_000)
    temperature_cdeg = 2500
    quality = int(np.clip(float(r["of_quality"]), 0, 255))
    distance_m = float(r["range_m"]) if np.isfinite(r["range_m"]) else 0.0

    master.mav.optical_flow_rad_send(
        time_usec,
        sensor_id,
        integration_time_us,
        float(r["of_integrated_x_rad"]),
        float(r["of_integrated_y_rad"]),
        float(r["of_integrated_xgyro_rad"]),
        float(r["of_integrated_ygyro_rad"]),
        float(r["of_integrated_zgyro_rad"]),
        temperature_cdeg,
        quality,
        distance_m,
    )


def replay(csv_path: str | Path, connection: str, baud: int, speed: float) -> None:
    df = pd.read_csv(csv_path)
    master = mavutil.mavlink_connection(connection, baud=baud)
    print("Ожидание heartbeat...")
    master.wait_heartbeat()
    print(f"Heartbeat: system={master.target_system}, component={master.target_component}")

    last_t = float(df.iloc[0]["t_s"])
    for _, r in df.iterrows():
        t_s = float(r["t_s"])
        dt = max(0.0, t_s - last_t)
        if speed > 0:
            time.sleep(dt / speed)
        last_t = t_s

        send_distance_sensor(master, t_s, float(r["range_m"]) if np.isfinite(r["range_m"]) else np.nan)
        send_optical_flow_rad(master, r)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay синтетического ОП-лога в MAVLink/SITL")
    parser.add_argument("--csv", default="data/synthetic_oflow_log.csv")
    parser.add_argument("--connection", default="udpout:127.0.0.1:14540")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--speed", type=float, default=1.0, help="1.0 = реальное время, 10 = быстрее")
    args = parser.parse_args()
    replay(args.csv, args.connection, args.baud, args.speed)


if __name__ == "__main__":
    main()
