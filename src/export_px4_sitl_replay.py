"""Read-only offline PX4 SITL diagnostic feed built from actual image VO.

It extracts accepted VO measurements from the existing synthetic_camera_imu
benchmark, pairs exact-time gyro and accel samples from the SAME generated
scenario, converts N/E velocity to body FRD then to the optical flow radian
convention already tested by the C++ ofnav estimator.

This is a diagnostic replay of the C++ algorithm; it does NOT publish
sensor_optical_flow, vehicle_imu, vehicle_odometry or actuator topics.
No model truth is read, and no flight-control commands are generated.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


FIELDS = (
    "timestamp_us", "arrival_us", "height_m", "yaw_rad",
    "gyro_x_rad_s", "gyro_y_rad_s", "gyro_z_rad_s",
    "accel_x_m_s2", "accel_y_m_s2", "accel_z_m_s2",
    "flow_x_rad", "flow_y_rad", "integration_dt_s",
    "quality", "flow_valid",
)


def _read(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as fp:
        reader = csv.DictReader(fp)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Отсутствуют столбцы {sorted(missing)} в {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Пустой журнал {path}")
    return rows


def export_replay(
    results_csv: str | Path, camera_csv: str | Path,
    imu_csv: str | Path, output_csv: str | Path,
) -> dict:
    results = _read(Path(results_csv), {
        "timestamp_us", "arrival_timestamp_imu_us",
        "visual_velocity_accepted", "vn_vo_m_s", "ve_vo_m_s",
    })
    frames = _read(Path(camera_csv), {
        "timestamp_us", "height_agl_m", "yaw_rad", "available",
        "arrival_timestamp_imu_us",
    })
    imus = _read(Path(imu_csv), {
        "timestamp_us", "gyro_x_rad_s", "gyro_y_rad_s", "gyro_z_rad_s",
        "accel_x_m_s2", "accel_y_m_s2", "accel_z_m_s2",
    })
    frame_by_t = {int(r["timestamp_us"]): r for r in frames}
    imu_by_t = {int(r["timestamp_us"]): r for r in imus}
    if len(frame_by_t) != len(frames) or len(imu_by_t) != len(imus):
        raise ValueError("Повтор временной метки исходного журнала")
    last = 0
    camera_stamps = sorted(frame_by_t)
    exported = []
    invalid = 0
    for row in results:
        t = int(row["timestamp_us"])
        arrival = int(row["arrival_timestamp_imu_us"])
        if t <= last or arrival < t:
            raise ValueError("Неверные временные метки визуального измерения")
        last = t
        if t not in frame_by_t:
            raise ValueError("Нет соответствующего кадра камеры")
        frame = frame_by_t[t]
        if int(frame["arrival_timestamp_imu_us"]) != arrival:
            raise ValueError("Время получения камеры изменено в результате VO")
        # Only real IMU samples from the same analytic scenario can serve
        # the replay. We do not silently interpolate or fill IMU gaps.
        imu = imu_by_t.get(t)
        valid = int(row["visual_velocity_accepted"]) == 1
        if valid and (imu is None or int(frame["available"]) != 1):
            raise ValueError("Принята VO без ИИМ/доступного кадра той же метки")
        if imu is None:
            # A gap is a diagnostic no-data frame. Use explicit invalid
            # markers without inventing a gyro or acceleration measurement.
            # This path is deliberately not accepted by the C++ replay.
            gyro_accel = [float("nan")] * 6
        else:
            gyro_accel = [
                float(imu[k]) for k in (
                    "gyro_x_rad_s", "gyro_y_rad_s", "gyro_z_rad_s",
                    "accel_x_m_s2", "accel_y_m_s2", "accel_z_m_s2",
                )
            ]
        height = float(frame["height_agl_m"])
        yaw = float(frame["yaw_rad"])
        if not (math.isfinite(height) and height > 0 and math.isfinite(yaw)):
            raise ValueError("Недопустимая модель высоты/курса")
        dt = 0.0
        if valid:
            # VO velocity belongs to the last observed frame pair in its
            # independent segment. After a blackout, use the preceding
            # image timestamp, never the last *accepted velocity* timestamp.
            previous_stamps = [stamp for stamp in camera_stamps if stamp < t]
            if not previous_stamps:
                raise ValueError("Нет начального кадра для интеграла потока")
            previous_stamp = previous_stamps[-1]
            if int(frame_by_t[previous_stamp]["available"]) != 1:
                raise ValueError("Принят VO после недоступного предыдущего кадра")
            dt = (t - previous_stamp) * 1e-6
            if not 0 < dt <= 0.2 + 1e-8:
                raise ValueError("Неподдерживаемый интервал интеграции оптического потока")
            vn, ve = float(row["vn_vo_m_s"]), float(row["ve_vo_m_s"])
            if not all(map(math.isfinite, (vn, ve))):
                raise ValueError("Принятый поток содержит неконечную скорость")
            body_x = math.cos(yaw)*vn + math.sin(yaw)*ve
            body_y = -math.sin(yaw)*vn + math.cos(yaw)*ve
            integrated_x = -(body_y / height) * dt
            integrated_y = (body_x / height) * dt
        else:
            integrated_x = integrated_y = 0.0
            dt = 0.0
            invalid += 1
        exported.append([
            t, arrival, height, yaw, *gyro_accel,
            integrated_x, integrated_y, dt,
            220 if valid else 0, int(valid),
        ])
    destination = Path(output_csv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(FIELDS)
        writer.writerows(exported)
    return {
        "rows": len(exported),
        "flow_valid_rows": len(exported)-invalid,
        "flow_invalid_rows": invalid,
        "source": "OFFLINE_SYNTHETIC_VO_AND_IMU",
        "publish_to_uorb": False,
        "publish_to_ekf2": False,
        "path": str(destination),
    }


def main() -> None:
    p=argparse.ArgumentParser(description="Безопасный CSV для replay в PX4 SITL без uORB-публикации")
    p.add_argument("--results-csv", required=True)
    p.add_argument("--camera-csv", required=True)
    p.add_argument("--imu-csv", required=True)
    p.add_argument("--output-csv", required=True)
    a=p.parse_args()
    print(export_replay(a.results_csv,a.camera_csv,a.imu_csv,a.output_csv))


if __name__=="__main__":
    main()
