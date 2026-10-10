"""Офлайн-воспроизведение реальной задержки поступления визуальной скорости.

Имитирует доступность сырых отсчетов ИИМ на их временной метке, а
визуального измерения — на arrival_timestamp_imu_us в той же шкале
часов. Время получения НЕ подменяет время середины экспозиции.
"""
from __future__ import annotations

import argparse
import csv
import json
from math import isfinite
from collections import Counter
from pathlib import Path

import yaml

from .error_state_inertial_filter import ImuNoiseDensity, InertialErrorStateFilter
from .fixed_lag_eskf_replay import FixedLagEskfReplay
from .horizontal_visual_inertial_filter import VisualVelocityMeasurement
from .imu_preintegration import ImuBias
from .run_imu_visual_fusion import _read_imu, _read_visual


def run_delayed_replay(
    imu_log: str | Path,
    visual_log: str | Path,
    config_yaml: str | Path,
    output_dir: str | Path,
) -> dict:
    with Path(config_yaml).open(encoding="utf-8") as f:
        settings = yaml.safe_load(f)
    if not isinstance(settings, dict) or (
        settings.get("stationary_calibration_confirmed") is not True
    ):
        raise ValueError("Нужна подтвержденная начальная калибровка ИИМ")
    if (settings.get("allow_experimental_visual_covariance") is not True
            and settings.get("visual_covariance_validated") is not True):
        raise ValueError("Нужна явная маркировка происхождения ковариации")
    try:
        noise = ImuNoiseDensity(
            float(settings["gyro_rad_s_sqrt_hz"]),
            float(settings["accel_m_s2_sqrt_hz"]),
            float(settings["gyro_bias_walk_rad_s2_sqrt_hz"]),
            float(settings["accel_bias_walk_m_s3_sqrt_hz"]),
        )
        bias = ImuBias(
            tuple(float(v) for v in settings["gyro_bias_body_rad_s"]),
            tuple(float(v) for v in settings["accel_bias_body_m_s2"]),
        )
        attitude = tuple(float(v) for v in settings["initial_quaternion_body_to_ned"])
        offset = int(settings.get("visual_to_imu_offset_us", 0))
        max_lag_us = int(settings.get("max_visual_lag_us", 300000))
        max_samples = int(settings.get("max_buffered_imu_samples", 1000))
        dt = float(settings.get("max_imu_dt_s", 0.05))
        def factory():
            return InertialErrorStateFilter(
                quaternion_body_to_ned=attitude, initial_bias=bias,
                noise=noise, max_imu_dt_s=dt,
            )
        engine = FixedLagEskfReplay(factory, max_lag_us=max_lag_us,
                                     max_imu_samples=max_samples)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Неполная или недопустимая конфигурация РФК") from exc

    imu = _read_imu(Path(imu_log))
    visuals = _read_visual(Path(visual_log))
    if not all("arrival_timestamp_imu_us" in row for row in visuals):
        raise ValueError("Журнал камеры должен содержать arrival_timestamp_imu_us")

    events = [(item.timestamp_us, 0, i, item)
              for i, item in enumerate(imu)]
    for i, item in enumerate(visuals):
        try:
            arrival = int(item["arrival_timestamp_imu_us"])
            exposure = int(item["timestamp_us"]) + offset
        except (ValueError, TypeError) as exc:
            raise ValueError("Неверные временные метки камеры") from exc
        if exposure <= 0 or arrival < exposure:
            raise ValueError("Кадр получен раньше экспозиции или неверная шкала часов")
        events.append((arrival, 1, i, item))
    events.sort(key=lambda e: (e[0], e[1], e[2]))

    stats = Counter()
    samples: list[dict] = []
    innovation_nis: list[float] = []
    for arrival, kind, _, obj in events:
        if kind == 0:
            engine.push_imu(obj)
            continue
        cam_time = int(obj["timestamp_us"])
        stamp = cam_time + offset
        if obj["accepted"] != "1":
            status, replayed, nis = "CAMERA_MEASUREMENT_REJECTED", 0, float("nan")
        else:
            try:
                covariance = (
                    (float(obj["velocity_cov_nn_m2_s2"]),
                     float(obj["velocity_cov_ne_m2_s2"])),
                    (float(obj["velocity_cov_ne_m2_s2"]),
                     float(obj["velocity_cov_ee_m2_s2"])),
                )
                measurement = VisualVelocityMeasurement(
                    stamp, float(obj["vn_m_s"]), float(obj["ve_m_s"]),
                    covariance
                )
                outcome = engine.push_delayed_visual(measurement)
                status, replayed, nis = (
                    outcome.status, outcome.replayed_imu_samples, outcome.nis
                )
            except (TypeError, ValueError):
                status, replayed, nis = "INVALID_VISUAL_FIELDS", 0, float("nan")
        snap = engine.state
        if status in ("VISUAL_CORRECTED", "VISUAL_OUTLIER_REJECTED") and (
            isfinite(nis)
        ):
            innovation_nis.append(nis)
        stats[status] += 1
        samples.append({
            "exposure_timestamp_imu_us": stamp,
            "arrival_timestamp_imu_us": arrival,
            "arrival_lag_us": arrival - stamp,
            "status": status,
            "replayed_imu_samples": replayed,
            "innovation_nis": nis,
            "current_imu_timestamp_us": snap.timestamp_us,
            "relative_n_m": snap.position_ned_m[0],
            "relative_e_m": snap.position_ned_m[1],
            "vn_m_s": snap.velocity_ned_m_s[0],
            "ve_m_s": snap.velocity_ned_m_s[1],
            "accepted_visual_count": snap.accepted_visual_count,
            "rejected_visual_count": snap.rejected_visual_count,
        })
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "delayed_visual_replay.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        fieldnames = list(samples[0]) if samples else ["exposure_timestamp_imu_us"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(samples)
    snap = engine.state
    summary = {
        "raw_imu_samples": len(imu),
        "visual_samples": len(visuals),
        "accepted_visual_updates": snap.accepted_visual_count,
        "nis_rejected_visual_updates": snap.rejected_visual_count,
        "statuses": dict(sorted(stats.items())),
        "covariance_policy": "EXPERIMENTAL_NOT_VALIDATED"
            if settings.get("allow_experimental_visual_covariance") else
            "USER_DECLARED_VALIDATED",
        "final_time_us": snap.timestamp_us,
        "history_max_lag_us": max_lag_us,
        "max_imu_storage": max_samples,
        "innovation_nis_samples": len(innovation_nis),
        "innovation_nis_mean_before_gating": (
            sum(innovation_nis) / len(innovation_nis) if innovation_nis else None
        ),
        "innovation_nis_fraction_above_99pct_chi2_2": (
            sum(x > 9.210340371976184 for x in innovation_nis) /
            len(innovation_nis) if innovation_nis else None
        ),
        "warnings": [
            "Офлайн-модель: полученные данные ИИМ предполагаются доступными в момент timestamp_us.",
            "Время прихода кадра не является временем визуального измерения.",
            "Абсолютное положение и карта отсутствуют, управление PX4 не подключено.",
            "Необходимо подтвердить шумы и статистическую согласованность по данным НИИВК.",
        ],
    }
    with (output / "delayed_visual_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Офлайн-проверка запаздывающих визуальных измерений РФК"
    )
    parser.add_argument("--imu-log", required=True)
    parser.add_argument("--visual-log", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(run_delayed_replay(
        args.imu_log, args.visual_log, args.config, args.output_dir
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
