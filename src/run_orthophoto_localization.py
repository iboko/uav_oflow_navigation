"""Reproducible offline evaluation of camera-to-orthophoto candidates.

YAML and CSV inputs declare metric local N/E coordinates and camera
calibration. Ground truth, if supplied, is NEVER used to select a match.
Output must not be sent to ESKF/PX4 without independently validated map
covariances and explicitly modeled shared-camera error correlations.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

from .camera_rotation_compensation import CameraMountCalibration
from .ground_visual_motion import CameraCalibration
from .map_correction_gate import assess_map_correction
from .map_temporal_integrity import TemporalMapIntegrityMonitor
from .orthophoto_localization import (
    MapGeoReference, OrthophotoTile, OrthophotoLocalizer
)

_REQUIRED = {
    "timestamp_us", "image_path", "height_agl_m",
    "roll_rad", "pitch_rad", "yaw_rad",
}
_PREDICTION = {
    "predicted_n_m", "predicted_e_m",
    "pred_cov_nn_m2", "pred_cov_ne_m2", "pred_cov_ee_m2",
}


def _load_system(config_file: Path) -> tuple[OrthophotoLocalizer, dict]:
    with config_file.open(encoding="utf-8") as f:
        settings = yaml.safe_load(f)
    if not isinstance(settings, dict):
        raise ValueError("Настройки камеры и листов карты должны быть словарем")
    try:
        c = settings["camera"]
        camera = CameraCalibration(
            float(c["focal_x_px"]), float(c["focal_y_px"]),
            float(c["center_x_px"]), float(c["center_y_px"]),
            tuple(tuple(float(x) for x in line) for line in c["image_to_body_xy"])
        )
        mount = CameraMountCalibration(tuple(
            tuple(float(x) for x in line) for line in c["body_from_camera"]
        ))
        tiles = []
        for item in settings["map_tiles"]:
            image_path = (config_file.parent / item["image_path"]).resolve()
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Не удалось открыть лист карты: {image_path}")
            geo = MapGeoReference(
                tuple(float(x) for x in item["origin_ne_m"]),
                tuple(tuple(float(x) for x in line) for line in
                      item["matrix_ne_m_per_px"]),
            )
            tiles.append(OrthophotoTile(str(item["name"]), image, geo))
        localizer = OrthophotoLocalizer(
            tiles, camera, mount,
            camera_offset_body_m=tuple(
                float(x) for x in c.get("camera_offset_body_m", (0.,0.,0.))
            )
        )
    except (TypeError, KeyError, ValueError) as exc:
        raise ValueError("Неполная или неверная калибровка карты/камеры") from exc
    return localizer, settings


def run_orthophoto_log(
    manifest: str | Path,
    configuration: str | Path,
    output_dir: str | Path,
) -> dict:
    manifest = Path(manifest)
    localizer, config = _load_system(Path(configuration))
    with manifest.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        columns = set(reader.fieldnames or [])
        if not _REQUIRED.issubset(columns):
            raise ValueError("Недостаточно обязательных полей видео CSV")
        if columns.intersection(_PREDICTION) and not _PREDICTION.issubset(columns):
            raise ValueError("Для проверки коррекции нужны все поля прогноза и ковариации")
        has_prediction = _PREDICTION.issubset(columns)
        samples = list(reader)
    if not samples:
        raise ValueError("Нет изображений для сопоставления")
    integrity_conf = config.get("temporal_integrity")
    temporal_monitor = None
    if integrity_conf is not None:
        if not isinstance(integrity_conf, dict):
            raise ValueError("Словарь temporal_integrity должен быть объектом YAML")
        try:
            temporal_monitor = TemporalMapIntegrityMonitor(
                max_horizontal_speed_m_s=float(
                    integrity_conf.get("max_horizontal_speed_m_s", 8.)
                ),
                max_interframe_gap_s=float(
                    integrity_conf.get("max_interframe_gap_s", .5)
                ),
                minimum_consistent_intervals=int(
                    integrity_conf.get("minimum_consistent_intervals", 3)
                ),
            )
        except (ValueError, TypeError) as exc:
            raise ValueError("Неверные настройки временной целостности карты") from exc
    rows = []
    statuses = Counter()
    gating = Counter()
    temporal_statuses = Counter()
    last_time = 0
    for index, row in enumerate(samples, 2):
        try:
            t = int(row["timestamp_us"])
            if t <= last_time:
                raise ValueError("Временные метки кадров должны возрастать")
            last_time = t
            image_path = (manifest.parent / row["image_path"]).resolve()
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Нет кадра: {image_path}")
            fixed = localizer.localize(
                image, height_agl_m=float(row["height_agl_m"]),
                roll_rad=float(row["roll_rad"]),
                pitch_rad=float(row["pitch_rad"]),
                yaw_rad=float(row["yaw_rad"]),
            )
            temporal_status = "NOT_EVALUATED"
            temporal_consistent = False
            if temporal_monitor is not None:
                temporal = temporal_monitor.inspect(t, fixed)
                temporal_status = temporal.status
                temporal_consistent = temporal.consistent
            gate_status = "NOT_REQUESTED_NO_ESTIMATOR_PRIOR"
            gate_nis = float("nan")
            if has_prediction:
                cov = config.get("map_covariance_ne_m2")
                prior = (
                    (float(row["pred_cov_nn_m2"]),float(row["pred_cov_ne_m2"])),
                    (float(row["pred_cov_ne_m2"]),float(row["pred_cov_ee_m2"])),
                )
                assessed = assess_map_correction(
                    fixed,
                    predicted_ne_m=(
                        float(row["predicted_n_m"]),float(row["predicted_e_m"])
                    ),
                    predicted_covariance_ne_m2=prior,
                    map_covariance_ne_m2=(
                        tuple(tuple(float(v) for v in line) for line in cov)
                        if cov is not None else None
                    ),
                    map_error_statistics_validated=(
                        config.get("map_error_statistics_validated") is True
                    ),
                    measurement_errors_independent_from_prior=(
                        config.get("measurement_errors_independent_from_prior")
                        is True
                    ),
                )
                gate_status, gate_nis = assessed.status, assessed.nis_2d
                # A declared validated covariance is not enough to use a
                # candidate in a sequence with impossible map jumps or without
                # the explicitly requested temporal support.
                if assessed.eligible and temporal_monitor is not None and (
                    not temporal_consistent
                ):
                    gate_status = "MAP_TEMPORAL_NOT_CONFIRMED"
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Ошибка строки видеолога {index}: {exc}") from exc
        statuses[fixed.status] += 1
        gating[gate_status] += 1
        temporal_statuses[temporal_status] += 1
        rows.append({
            "timestamp_us": t,
            "image_path": row["image_path"],
            "map_status": fixed.status,
            "matched_tile": fixed.tile_name,
            "candidate_n_m": fixed.north_m,
            "candidate_e_m": fixed.east_m,
            "matched_points": fixed.matched_points,
            "inlier_points": fixed.inlier_points,
            "inlier_fraction": fixed.inlier_fraction,
            "median_residual_map_px": fixed.median_residual_map_px,
            "jacobian_error_fraction": fixed.geometric_jacobian_error,
            "map_gate_status": gate_status,
            "temporal_integrity_status": temporal_status,
            "temporal_integrity_only_unvalidated": True,
            "map_gate_nis_2d": gate_nis,
            "applied_to_estimator": 0,
        })

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    with (target / "orthophoto_candidates.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "frames": len(rows),
        "geometrically_accepted": sum(
            r["map_status"] == "GEOMETRICALLY_ACCEPTED_UNVALIDATED" for r in rows
        ),
        "map_status_counts": dict(sorted(statuses.items())),
        "correction_gate_counts": dict(sorted(gating.items())),
        "temporal_integrity_counts": dict(sorted(temporal_statuses.items())),
        "applied_to_estimator": 0,
        "remarks": [
            "Предложение положения БВС в локальных метрах N/E, не абсолютные широта/долгота.",
            "Ковариация карты определяется по независимым испытаниям; геометрический остаток не заменяет ее.",
            "При общих изображениях VO/карты ошибки коррелируют: необходимо учитывать перекрестную ковариацию.",
            "Только плоская местность и предварительно устраненная дисторсия камеры.",
            "Ни алгоритм удержания позиции, ни интерфейс управления PX4 здесь не реализованы.",
        ],
    }
    with (target / "orthophoto_summary.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Независимая офлайн-привязка нижней камеры БВС к ортофотоплану"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--configuration", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(run_orthophoto_log(
        args.manifest, args.configuration, args.output_dir,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
