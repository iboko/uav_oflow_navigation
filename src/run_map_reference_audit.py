"""Independent reference audit of map localization candidates (offline only).

A geometrically accepted frame can be a false global map match. The
false-fix threshold is an externally specified operational tolerance,
not derived from the feature-matching residual. Truth is NEVER passed
back into the locator or ESKF. Per-flight aggregation keeps frame
correlation distinct from the number of independent sorties.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


def _rows(path: Path, required: set[str]) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Отсутствуют поля {sorted(missing)} в {path}")
        data = list(reader)
    if not data:
        raise ValueError(f"Пустой журнал {path}")
    return data


def _indexed(rows: list[dict]) -> dict[int, dict]:
    out = {}
    last = 0
    for row in rows:
        try:
            t = int(row["timestamp_us"])
        except (TypeError, ValueError) as exc:
            raise ValueError("Некорректная временная метка") from exc
        if t <= last or t <= 0:
            raise ValueError("Метки времени должны строго возрастать")
        last = t
        out[t] = row
    return out


def _reference(row: dict) -> np.ndarray:
    label = row.get("source", "").strip().lower()
    if not label or label in ("filter", "estimator", "self", "map_localizer"):
        raise ValueError("Требуется независимая эталонная измерительная система")
    ne = np.array([float(row["true_n_m"]), float(row["true_e_m"])], dtype=float)
    if not np.isfinite(ne).all():
        raise ValueError("Неконечные независимые эталонные координаты")
    return ne


def evaluate_flight(
    candidates: list[dict],
    reference: list[dict],
    *,
    false_fix_error_threshold_m: float,
) -> tuple[dict, list[dict]]:
    if not np.isfinite(false_fix_error_threshold_m) or (
        not 0 < false_fix_error_threshold_m <= 100
    ):
        raise ValueError("Задайте положительный предел ложной привязки в метрах")
    cand = _indexed(candidates)
    truth = _indexed(reference)
    if not set(cand).issubset(truth):
        raise ValueError("Не у всех кадров есть независимый эталон с той же меткой")
    all_rows = []
    distances = []
    false_fixes = 0
    accepted = 0
    for t, result in cand.items():
        reference_ne = _reference(truth[t])
        geometric = result["map_status"] == "GEOMETRICALLY_ACCEPTED_UNVALIDATED"
        residual_m = None
        if geometric:
            estimate = np.array(
                [float(result["candidate_n_m"]), float(result["candidate_e_m"])],
                dtype=float
            )
            if not np.isfinite(estimate).all():
                raise ValueError("Принятая геометрия не содержит конечных координат")
            residual_m = float(np.linalg.norm(estimate-reference_ne))
            distances.append(residual_m)
            accepted += 1
            false_fixes += residual_m > false_fix_error_threshold_m
        all_rows.append({
            "timestamp_us": t,
            "geometric_accept": int(geometric),
            "error_ne_m": residual_m,
            "false_fix": int(
                residual_m is not None and residual_m > false_fix_error_threshold_m
            ),
            "map_status": result["map_status"],
        })
    errors = np.asarray(distances, dtype=float)
    return {
        "frames_evaluated": len(cand),
        "geometrically_accepted": accepted,
        "false_fixes": false_fixes,
        "false_fix_error_threshold_m": false_fix_error_threshold_m,
        "false_fix_rate_among_accepted": (
            false_fixes/accepted if accepted else None
        ),
        "false_fix_fraction_of_all_frames": false_fixes/len(cand),
        "geometric_acceptance_fraction": accepted/len(cand),
        "rmse_ne_m_on_accepted": (
            float(np.sqrt(np.mean(errors**2))) if accepted else None
        ),
        "median_error_ne_m_on_accepted": (
            float(np.median(errors)) if accepted else None
        ),
        "p95_error_ne_m_on_accepted": (
            float(np.percentile(errors, 95)) if accepted else None
        ),
        "truth_fraction_of_estimate_frames": 1.0,
        "warning": "Синтетические и автокоррелированные кадры не подтверждают натурную точность",
    }, all_rows


def run_reference_audit(
    flight_manifest: str | Path,
    output_dir: str | Path,
    *,
    false_fix_error_threshold_m: float,
) -> dict:
    manifest = Path(flight_manifest)
    flights = _rows(manifest, {"flight_id", "candidate_csv", "truth_csv"})
    if len({row["flight_id"] for row in flights}) != len(flights):
        raise ValueError("Идентификаторы полетов должны быть уникальны")
    result = {}
    per_frame = []
    for record in flights:
        name = record["flight_id"].strip()
        if not name:
            raise ValueError("Пустой идентификатор полета")
        candidate_rows = _rows(
            manifest.parent / record["candidate_csv"],
            {"timestamp_us", "map_status", "candidate_n_m", "candidate_e_m"},
        )
        truth_rows = _rows(
            manifest.parent / record["truth_csv"],
            {"timestamp_us", "true_n_m", "true_e_m", "source"},
        )
        summary, matched = evaluate_flight(
            candidate_rows, truth_rows,
            false_fix_error_threshold_m=false_fix_error_threshold_m
        )
        result[name] = summary
        per_frame.extend({"flight_id": name, **entry} for entry in matched)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output/"map_reference_errors.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_frame[0].keys()))
        writer.writeheader()
        writer.writerows(per_frame)
    report = {
        "independent_flights": len(result),
        "flights": result,
        "all_flights_had_independent_truth": True,
        "mean_false_fix_rate_over_flights": (
            float(np.mean([
                v["false_fix_rate_among_accepted"]
                for v in result.values()
                if v["false_fix_rate_among_accepted"] is not None
            ])) if any(v["geometrically_accepted"] for v in result.values()) else None
        ),
        "validated_for_flight": False,
        "limitations": [
            "Информацию о независимости и калибровке эталона подтверждает испытательная группа",
            "Проверка выполняется только при точном совпадении временных меток",
            "Доля ошибочных принятых привязок зависит от заданного эксплуатационного порога",
            "Корреляция кадров внутри одного полета не позволяет использовать их как независимые испытания",
            "Численное подтверждение точности на 100 м требует реальных полетных записей",
        ],
    }
    with (output/"map_reference_summary.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Оценка ложных принятых картографических привязок по независимому эталону"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--false-fix-error-threshold-m", required=True, type=float)
    args = parser.parse_args()
    print(json.dumps(run_reference_audit(
        args.manifest, args.output_dir,
        false_fix_error_threshold_m=args.false_fix_error_threshold_m
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
