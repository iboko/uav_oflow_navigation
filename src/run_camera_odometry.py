"""Воспроизводимая офлайн-обработка кадров и расчет ошибок относительно эталона.

Запуск: python -m src.run_camera_odometry --manifest log.csv
        --calibration camera.yaml --output-dir out/

Эталон НЕ участвует в оценивании; используется только для проверки
относительных смещений в каждом наблюдаемом участке.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from math import hypot, isfinite, sqrt
from pathlib import Path

import cv2
import numpy as np
import yaml

from .continuous_visual_odometry import ContinuousPlanarOdometry, VisualFrame
from .ground_visual_motion import CameraCalibration

_REQUIRED = (
    "timestamp_us", "image_path", "height_agl_m",
    "roll_rad", "pitch_rad", "yaw_rad",
)


def _camera(path: str | Path) -> CameraCalibration:
    with Path(path).open(encoding="utf-8") as stream:
        src = yaml.safe_load(stream)
    if not isinstance(src, dict):
        raise ValueError("Ожидается словарь параметров калибровки")
    try:
        calib = CameraCalibration(
            focal_x_px=float(src["focal_x_px"]),
            focal_y_px=float(src["focal_y_px"]),
            center_x_px=float(src["center_x_px"]),
            center_y_px=float(src["center_y_px"]),
            image_to_body_xy=tuple(
                tuple(float(v) for v in row) for row in src["image_to_body_xy"]
            ),
        )
    except (ValueError, TypeError, KeyError) as exc:
        raise ValueError("Неполная или некорректная калибровка камеры") from exc
    if not calib.valid():
        raise ValueError("Недопустимая калибровка: проверьте фокус и оси камеры")
    return calib


def run_manifest(
    manifest: str | Path, calibration: str | Path, output_dir: str | Path,
) -> dict:
    manifest = Path(manifest)
    output_dir = Path(output_dir)
    odom = ContinuousPlanarOdometry(_camera(calibration))

    rows: list[dict] = []
    reasons: Counter[str] = Counter()
    origin_by_segment: dict[int, tuple[float, float]] = {}
    errors: list[float] = []

    with manifest.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError("Пустой или недопустимый перечень кадров")
        missing = sorted(set(_REQUIRED) - set(reader.fieldnames))
        if missing:
            raise ValueError(f"Отсутствуют обязательные столбцы: {missing}")
        truth_columns = {"true_n_m", "true_e_m"}
        if truth_columns.intersection(reader.fieldnames) and not truth_columns.issubset(reader.fieldnames):
            raise ValueError("Эталон требует одновременно true_n_m и true_e_m")
        has_truth = truth_columns.issubset(reader.fieldnames)

        for line_no, record in enumerate(reader, start=2):
            name = record["image_path"]
            file_path = (manifest.parent / name).resolve()
            gray = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise ValueError(f"Строка {line_no}: не удалось прочитать кадр {file_path}")

            try:
                sample = VisualFrame(
                    int(record["timestamp_us"]), gray, float(record["height_agl_m"]),
                    float(record["yaw_rad"]), float(record["roll_rad"]),
                    float(record["pitch_rad"]),
                )
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Строка {line_no}: неверные числовые данные") from exc

            result = odom.process(sample)
            reasons[result.status] += 1
            current = {
                "timestamp_us": sample.timestamp_us,
                "image_path": name,
                "segment_id": result.segment_id,
                "status": result.status,
                "accepted": int(result.accepted),
                "rel_n_m": result.north_m,
                "rel_e_m": result.east_m,
                "vn_m_s": result.vn_m_s,
                "ve_m_s": result.ve_m_s,
                "inlier_count": result.inlier_count,
                "inlier_ratio": result.inlier_ratio,
                "resolution_proxy_m": result.resolution_proxy_m,
                "relative_position_error_m": float("nan"),
            }

            if has_truth:
                try:
                    tn, te = float(record["true_n_m"]), float(record["true_e_m"])
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"Строка {line_no}: неверный эталон") from exc
                if not isfinite(tn) or not isfinite(te):
                    raise ValueError(f"Строка {line_no}: эталон не конечен")
                # Every uninterrupted segment gets its own reference origin;
                # comparing against absolute global truth would be misleading.
                if result.status in ("INITIALIZED", "TIME_GAP_REINITIALIZED",
                                     "IMAGE_SIZE_CHANGED", "HEIGHT_JUMP",
                                     "EXCESSIVE_YAW_CHANGE") or result.status.startswith("TRACK_LOST:"):
                    origin_by_segment[result.segment_id] = (tn, te)
                if result.accepted and result.segment_id in origin_by_segment:
                    n0, e0 = origin_by_segment[result.segment_id]
                    error = hypot(result.north_m - (tn - n0),
                                  result.east_m - (te - e0))
                    errors.append(error)
                    current["relative_position_error_m"] = error
            rows.append(current)

    if not rows:
        raise ValueError("В перечне нет кадров")

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "visual_odometry.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    valid = sum(int(row["accepted"]) for row in rows)
    report = {
        "frames": len(rows),
        "accepted_intervals": valid,
        "accepted_fraction_per_frame": valid / len(rows),
        "independent_segments": len(set(row["segment_id"] for row in rows)),
        "status_counts": dict(sorted(reasons.items())),
        "reference_available": has_truth,
        "reference_evaluated_intervals": len(errors),
        # Error assessed only on observed intervals, relative to segment start.
        "relative_position_rmse_m": (
            sqrt(sum(v * v for v in errors) / len(errors)) if errors else None
        ),
        "relative_position_p95_m": (
            float(np.percentile(errors, 95)) if errors else None
        ),
        "warning": (
            "Относительная двухмерная одометрия при модели плоской поверхности; "
            "между сегментами координатной привязки нет. "
            "Индикатор разрешения не является доверительным интервалом."
        ),
    }
    with (output_dir / "visual_odometry_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Проверка визуальной одометрии БВС по видеологу")
    parser.add_argument("--manifest", required=True, help="CSV перечень кадров")
    parser.add_argument("--calibration", required=True, help="YAML параметры камеры")
    parser.add_argument("--output-dir", default="outputs/visual_odometry", help="Каталог результатов")
    args = parser.parse_args()
    result = run_manifest(args.manifest, args.calibration, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
