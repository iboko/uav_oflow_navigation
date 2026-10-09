"""Воспроизводимая офлайн-проверка 15-мерного инерциального РФК.

Получает сырые ИИМ и относительную горизонтальную визуальную скорость.
Никаких команд PX4 и никакой географической привязки. Измерения
скорости с неподтвержденной ковариацией допускаются лишь по явному
разрешению ДЛЯ ИССЛЕДОВАТЕЛЬСКОГО РЕЖИМА, не для управления БВС.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from math import isfinite
from pathlib import Path

import numpy as np
import yaml

from .error_state_inertial_filter import (
    ImuNoiseDensity, InertialErrorStateFilter
)
from .horizontal_visual_inertial_filter import VisualVelocityMeasurement
from .imu_preintegration import ImuBias
from .run_imu_visual_fusion import _read_imu, _read_visual, _interpolate_imu


def run_eskf_offline(
    imu_log: str | Path,
    visual_log: str | Path,
    config_yaml: str | Path,
    output_dir: str | Path,
) -> dict:
    with Path(config_yaml).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Ожидается словарь конфигурации")
    if config.get("stationary_calibration_confirmed") is not True:
        raise ValueError("Не подтверждена начальная калибровка ИИМ")
    verified = config.get("visual_covariance_validated") is True
    experimental = config.get("allow_experimental_visual_covariance") is True
    if not verified and not experimental:
        raise ValueError("Нужна подтвержденная ковариация либо явное разрешение на эксперимент")
    if verified and experimental:
        raise ValueError("Выберите один режим ковариации, не оба")
    try:
        bias = ImuBias(
            tuple(float(v) for v in config["gyro_bias_body_rad_s"]),
            tuple(float(v) for v in config["accel_bias_body_m_s2"]),
        )
        noise = ImuNoiseDensity(
            **{name: float(config[name]) for name in (
                "gyro_rad_s_sqrt_hz", "accel_m_s2_sqrt_hz",
                "gyro_bias_walk_rad_s2_sqrt_hz",
                "accel_bias_walk_m_s3_sqrt_hz",
            )}
        )
        eskf = InertialErrorStateFilter(
            quaternion_body_to_ned=tuple(
                float(v) for v in config["initial_quaternion_body_to_ned"]
            ),
            initial_bias=bias,
            noise=noise,
            max_imu_dt_s=float(config.get("max_imu_dt_s", 0.05)),
        )
        offset = int(config.get("visual_to_imu_offset_us", 0))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Некорректная конфигурация ИИМ/ESKF") from exc

    imu = _read_imu(Path(imu_log))
    visual = _read_visual(Path(visual_log))
    latest_raw = 0
    outcomes: list[dict] = []
    reasons = Counter()
    nis_values = []
    last_visual_time = 0
    for row in visual:
        camera_time = int(row["timestamp_us"])
        t = camera_time + offset
        if t <= last_visual_time or t <= 0:
            raise ValueError("После пересчета часы видеокадров должны строго возрастать")
        last_visual_time = t
        while latest_raw < len(imu) and imu[latest_raw].timestamp_us <= t:
            eskf.predict(imu[latest_raw])
            latest_raw += 1
        reason = "VISUAL_NOT_ACCEPTED_BY_CAMERA"
        if row["accepted"] == "1":
            snap = eskf.snapshot()
            if t < imu[0].timestamp_us or t > imu[-1].timestamp_us:
                reason = "VISUAL_OUTSIDE_IMU_TIMELINE"
            elif snap.status == "IMU_INVALID_OR_GAP":
                reason = snap.status
            elif snap.timestamp_us < t:
                if latest_raw >= len(imu):
                    reason = "VISUAL_OUTSIDE_IMU_TIMELINE"
                elif (imu[latest_raw].timestamp_us -
                      imu[latest_raw - 1].timestamp_us) * 1e-6 > eskf.max_imu_dt_s:
                    reason = "IMU_GAP_AT_CAMERA_FRAME"
                else:
                    eskf.predict(_interpolate_imu(
                        imu[latest_raw - 1], imu[latest_raw], t
                    ))
                    if eskf.snapshot().status == "IMU_INVALID_OR_GAP":
                        reason = "IMU_INVALID_OR_GAP"
            if reason == "VISUAL_NOT_ACCEPTED_BY_CAMERA":
                try:
                    cov = (
                        (float(row["velocity_cov_nn_m2_s2"]),
                         float(row["velocity_cov_ne_m2_s2"])),
                        (float(row["velocity_cov_ne_m2_s2"]),
                         float(row["velocity_cov_ee_m2_s2"])),
                    )
                    measurement = VisualVelocityMeasurement(
                        t, float(row["vn_m_s"]), float(row["ve_m_s"]), cov
                    )
                    result = eskf.update_visual_velocity(measurement)
                    reason = result.status
                    if (reason in ("VISUAL_CORRECTED", "VISUAL_OUTLIER_REJECTED")
                            and isfinite(result.last_nis)):
                        nis_values.append(result.last_nis)
                except (TypeError, ValueError):
                    reason = "INVALID_VISUAL_FIELDS"
        snap = eskf.snapshot()
        reasons[reason] += 1
        outcomes.append({
            "camera_timestamp_us": camera_time,
            "imu_timestamp_us": snap.timestamp_us,
            "status": reason,
            "north_m": snap.position_ned_m[0],
            "east_m": snap.position_ned_m[1],
            "down_m": snap.position_ned_m[2],
            "vn_m_s": snap.velocity_ned_m_s[0],
            "ve_m_s": snap.velocity_ned_m_s[1],
            "vd_m_s": snap.velocity_ned_m_s[2],
            "qw": snap.quaternion_body_to_ned[0],
            "qx": snap.quaternion_body_to_ned[1],
            "qy": snap.quaternion_body_to_ned[2],
            "qz": snap.quaternion_body_to_ned[3],
            "gyro_bias_z_rad_s": snap.gyro_bias_rad_s[2],
            "accel_bias_x_m_s2": snap.accel_bias_m_s2[0],
            "covariance_vn_m2_s2": snap.covariance_15x15[3][3],
            "covariance_ve_m2_s2": snap.covariance_15x15[4][4],
            "innovation_nis": snap.last_nis,
        })
    while latest_raw < len(imu):
        eskf.predict(imu[latest_raw])
        latest_raw += 1
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "eskf_trajectory.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        fields = list(outcomes[0].keys()) if outcomes else ["camera_timestamp_us"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(outcomes)
    report = {
        "imu_samples": len(imu),
        "visual_samples": len(visual),
        "accepted_visual_updates": eskf.snapshot().accepted_visual_count,
        "rejected_visual_updates": eskf.snapshot().rejected_visual_count,
        "reasons": dict(sorted(reasons.items())),
        "final_status": eskf.snapshot().status,
        "covariance_mode": (
            "EXTERNAL_VALIDATION_DECLARED"
            if verified else "EXPERIMENTAL_UNVALIDATED"
        ),
        "nis_samples": len(nis_values),
        "nis_mean": float(np.mean(nis_values)) if nis_values else None,
        "nis_expected_mean_if_well_calibrated": 2.0,
        "limitations": [
            "Начало координат произвольное: нет географической привязки.",
            "Скорость VO двумерная; 15 состояний могут быть ненаблюдаемы.",
            "NIS=2 в среднем ожидается только при верной статистической модели.",
            "Нет повторного прогнозирования при поступлении запаздывающих кадров.",
            "Офлайн-алгоритм запрещено выдавать за источник PX4/EKF2.",
        ],
    }
    with (out / "eskf_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Воспроизводимая проверка 15-мерного РФК по логам НИИВК"
    )
    parser.add_argument("--imu-log", required=True)
    parser.add_argument("--visual-log", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(run_eskf_offline(
        args.imu_log, args.visual_log, args.config, args.output_dir
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
