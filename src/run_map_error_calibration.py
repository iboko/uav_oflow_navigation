"""Reproducible train/holdout calibration of map-fix error on separate flights.

Inputs must be external, independently measured in metres (local N/E).
No assumptions are made that synthetic results prove real accuracy.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .map_error_calibration import MapErrorFlight, calibrate_map_error
from .run_map_reference_audit import _rows, _indexed, _reference


def _flight_from_files(
    flight_id: str, candidates: Path, truth: Path,
    *, false_fix_error_threshold_m: float,
) -> MapErrorFlight:
    rows = _indexed(_rows(
        candidates, {"timestamp_us", "map_status", "candidate_n_m", "candidate_e_m"}
    ))
    references = _indexed(_rows(
        truth, {"timestamp_us", "source", "true_n_m", "true_e_m"}
    ))
    if not set(rows).issubset(references):
        raise ValueError("Пропущены независимые эталонные кадры")
    errors = []
    false_fixes = 0
    for t, row in rows.items():
        ref = _reference(references[t])
        if row["map_status"] != "GEOMETRICALLY_ACCEPTED_UNVALIDATED":
            continue
        position = np.array([
            float(row["candidate_n_m"]), float(row["candidate_e_m"])
        ], dtype=float)
        if not np.isfinite(position).all():
            raise ValueError("Принятая карта содержит неконечные координаты")
        error = position - ref
        errors.append(tuple(float(x) for x in error))
        false_fixes += float(np.linalg.norm(error)) > false_fix_error_threshold_m
    return MapErrorFlight(flight_id, tuple(errors), len(rows), false_fixes)


def run_map_calibration(
    manifest: str | Path,
    output_dir: str | Path,
    *,
    false_fix_error_threshold_m: float,
    min_samples_per_flight: int = 30,
) -> dict:
    if not np.isfinite(false_fix_error_threshold_m) or (
        false_fix_error_threshold_m <= 0
    ):
        raise ValueError("Задайте внешний положительный предел ошибки привязки")
    manifest_path = Path(manifest)
    rows = _rows(
        manifest_path, {"flight_id", "partition", "candidate_csv", "truth_csv"}
    )
    training, holdout = [], []
    for row in rows:
        role = row["partition"].strip().lower()
        if role not in ("train", "holdout"):
            raise ValueError("Каждый полет должен быть помечен train или holdout")
        flight = _flight_from_files(
            row["flight_id"].strip(),
            manifest_path.parent / row["candidate_csv"],
            manifest_path.parent / row["truth_csv"],
            false_fix_error_threshold_m=false_fix_error_threshold_m
        )
        (training if role == "train" else holdout).append(flight)
    estimate = calibrate_map_error(
        training, holdout, min_samples_per_flight=min_samples_per_flight
    )
    result = asdict(estimate)
    result["false_fix_error_threshold_m"] = false_fix_error_threshold_m
    result["covariance_model_status"] = "EMPIRICAL_HOLDOUT_UNCERTIFIED"
    result["flightworthiness_approved"] = False
    result["warnings"] = [
        "Оценка получена только по геометрически принятым кадрам; это условная ковариация",
        "Видеокадры в пределах одного полета коррелированы; число полетов является независимой единицей анализа",
        "Разделение train/holdout должно быть установлено до анализа и без повторного использования полетов",
        "Если систематическая ошибка зависит от высоты, карты, рельефа или времени суток, нужна раздельная модель",
        "Даже хорошая эмпирическая статистика не исключает невыявленные ложные сопоставления",
        "Модель НЕ автоматически допускается в ESKF/PX4 и не доказывает точности на высоте 100 м",
    ]
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    with (dest/"map_error_calibration.json").open("w",encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Раздельная оценка ошибок привязки карты по независимым полетам"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--false-fix-error-threshold-m", required=True, type=float)
    args = parser.parse_args()
    print(json.dumps(run_map_calibration(
        args.manifest, args.output_dir,
        false_fix_error_threshold_m=args.false_fix_error_threshold_m
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
