"""Офлайн-слияние визуальной скорости с сырыми измерениями ИИМ.

Не управляет БВС. Входной visual_odometry.csv формирует
src.run_camera_odometry, а IMU.csv — синхронный журнал ИИМ.
Нельзя подавать визуальную скорость с невалидированной/отсутствующей
матрицей ошибок в рабочий PX4 EKF2.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from .horizontal_visual_inertial_filter import (
    HorizontalVisualInertialFilter, VisualVelocityMeasurement
)
from .imu_preintegration import ImuBias, ImuReading

_IMU_FIELDS = (
    "timestamp_us", "gyro_x_rad_s", "gyro_y_rad_s", "gyro_z_rad_s",
    "accel_x_m_s2", "accel_y_m_s2", "accel_z_m_s2",
)
_VIS_FIELDS = (
    "timestamp_us", "accepted", "vn_m_s", "ve_m_s",
    "velocity_cov_nn_m2_s2", "velocity_cov_ne_m2_s2",
    "velocity_cov_ee_m2_s2",
)


def _read_imu(path: Path) -> list[ImuReading]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or not set(_IMU_FIELDS).issubset(reader.fieldnames):
            raise ValueError("Некорректные столбцы журнала ИИМ")
        readings = []
        for k, row in enumerate(reader, 2):
            try:
                readings.append(ImuReading(
                    int(row["timestamp_us"]),
                    tuple(float(row[f"gyro_{a}_rad_s"]) for a in "xyz"),
                    tuple(float(row[f"accel_{a}_m_s2"]) for a in "xyz"),
                ))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Строка ИИМ {k}: неверные значения") from exc
    if not readings or any(
        b.timestamp_us <= a.timestamp_us for a, b in zip(readings, readings[1:])
    ):
        raise ValueError("Журнал ИИМ пуст или время не возрастает")
    return readings


def _read_visual(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or not set(_VIS_FIELDS).issubset(reader.fieldnames):
            raise ValueError("Нужны визуальная скорость и полная горизонтальная ковариация")
        rows = list(reader)
    if any(int(row["timestamp_us"]) <= 0 for row in rows):
        raise ValueError("Недопустимые временные метки визуального лога")
    if any(int(b["timestamp_us"]) <= int(a["timestamp_us"]) for a, b in zip(rows, rows[1:])):
        raise ValueError("Визуальные временные метки должны возрастать")
    return rows


def _interpolate_imu(before: ImuReading, after: ImuReading, stamp: int) -> ImuReading:
    if not before.timestamp_us < stamp < after.timestamp_us:
        raise ValueError("Точка интерполяции вне интервала ИИМ")
    alpha = (stamp - before.timestamp_us) / (
        after.timestamp_us - before.timestamp_us
    )
    gyro = (1.0 - alpha) * np.asarray(before.gyro_body_rad_s) + (
        alpha * np.asarray(after.gyro_body_rad_s)
    )
    accel = (1.0 - alpha) * np.asarray(before.specific_force_body_m_s2) + (
        alpha * np.asarray(after.specific_force_body_m_s2)
    )
    return ImuReading(stamp, tuple(gyro), tuple(accel))


def run_fusion(
    imu_log: str | Path,
    visual_log: str | Path,
    config_yaml: str | Path,
    output_dir: str | Path,
) -> dict:
    with Path(config_yaml).open(encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    if not isinstance(cfg, dict) or cfg.get("stationary_calibration_confirmed") is not True:
        raise ValueError("Нужна подтвержденная начальная калибровка ИИМ")
    try:
        initial_bias = ImuBias(
            tuple(float(v) for v in cfg["gyro_bias_body_rad_s"]),
            tuple(float(v) for v in cfg["accel_bias_body_m_s2"]),
        )
        f = HorizontalVisualInertialFilter(
            quaternion_body_to_ned=tuple(
                float(v) for v in cfg["initial_quaternion_body_to_ned"]
            ),
            initial_bias=initial_bias,
            gyro_noise_rad_s=float(cfg["gyro_noise_rad_s"]),
            accel_noise_m_s2=float(cfg["accel_noise_m_s2"]),
            accel_bias_walk_m_s2_sqrt_s=float(cfg["accel_bias_walk_m_s2_sqrt_s"]),
            max_imu_dt_s=float(cfg.get("max_imu_dt_s", 0.05)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Недопустимая или неполная калибровка ИИМ/фильтра") from exc
    imu = _read_imu(Path(imu_log))
    visual = _read_visual(Path(visual_log))
    max_gap_us = int(f._imu.max_dt_s * 1e6)
    next_imu = 0
    results = []
    statuses = Counter()
    corrected = 0

    for row in visual:
        t = int(row["timestamp_us"])
        while next_imu < len(imu) and imu[next_imu].timestamp_us <= t:
            f.predict(imu[next_imu])
            next_imu += 1
        status = "VISUAL_NOT_ACCEPTED_BY_CAMERA"
        if row["accepted"] == "1":
            state = f.snapshot()
            if (t < imu[0].timestamp_us or t > imu[-1].timestamp_us
                    or not state.timestamp_us):
                status = "OUTSIDE_IMU_TIMELINE"
            elif state.timestamp_us < t:
                if next_imu >= len(imu):
                    status = "OUTSIDE_IMU_TIMELINE"
                elif (imu[next_imu].timestamp_us - imu[next_imu - 1].timestamp_us
                      > max_gap_us):
                    status = "IMU_GAP_AT_FRAME"
                else:
                    f.predict(_interpolate_imu(imu[next_imu - 1], imu[next_imu], t))
            if status == "VISUAL_NOT_ACCEPTED_BY_CAMERA":
                # status is unchanged when interpolation/IMU time was valid.
                try:
                    cov = (
                        (float(row["velocity_cov_nn_m2_s2"]),
                         float(row["velocity_cov_ne_m2_s2"])),
                        (float(row["velocity_cov_ne_m2_s2"]),
                         float(row["velocity_cov_ee_m2_s2"])),
                    )
                    observation = VisualVelocityMeasurement(
                        t, float(row["vn_m_s"]), float(row["ve_m_s"]), cov
                    )
                    status = f.update_visual(observation).status
                    corrected += (status == "VISUAL_CORRECTED")
                except (ValueError, TypeError) as exc:
                    status = "INVALID_VISUAL_FIELDS"
        state = f.snapshot()
        statuses[status] += 1
        results.append({
            "timestamp_us": t,
            "visual_status": status,
            "relative_north_m": state.north_m,
            "relative_east_m": state.east_m,
            "vn_m_s": state.vn_m_s,
            "ve_m_s": state.ve_m_s,
            "estimated_accel_bias_x_m_s2": state.accel_bias_x_m_s2,
            "estimated_accel_bias_y_m_s2": state.accel_bias_y_m_s2,
            "innovation_nis": state.nis,
            "visual_updates": state.visual_updates,
        })
    while next_imu < len(imu):
        f.predict(imu[next_imu])
        next_imu += 1
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "imu_visual_fusion.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0].keys())
                                if results else ["timestamp_us"])
        writer.writeheader()
        writer.writerows(results)
    report = {
        "imu_samples": len(imu),
        "visual_samples": len(visual),
        "accepted_visual_updates": corrected,
        "status_counts": dict(sorted(statuses.items())),
        "final_status": f.snapshot().status,
        "warnings": [
            "Относительная горизонтальная оценка; абсолютной геопривязки нет.",
            "Ковариация ошибки ориентации не учтена: результат не годится для EKF2.",
            "Запаздывающая визуальная скорость не перепропагируется — время должно быть согласовано.",
            "Коэффициенты ковариации и начальная калибровка должны подтверждаться НИИВК.",
        ],
    }
    with (out / "imu_visual_summary.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Офлайн-слияние визуальной скорости с сырыми данными ИИМ"
    )
    parser.add_argument("--imu-log", required=True)
    parser.add_argument("--visual-log", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(run_fusion(
        args.imu_log, args.visual_log, args.config, args.output_dir
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
