"""Offline ESKF consistency review against INDEPENDENT reference flights.

Usage:
 python -m src.run_eskf_consistency --manifest validation/flights.csv \
       --output-dir outputs/consistency

Manifest columns:
 flight_id,states_jsonl,truth_csv,nis_csv,linearization_csv,
 origin_alignment_verified,heading_alignment_verified

Only the first three are mandatory. Optional paths may be blank.
Paths are relative to manifest directory. NO GROUND TRUTH enters estimator.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .error_state_inertial_filter import EskfSnapshot
from .eskf_consistency import (
    LinearizationSample, TruthSample, flight_level_summary,
    nees_full_15, nees_horizontal_velocity, observability_velocity_only,
    quaternion_angular_error_rad, CHI2_2_99,
)
from .imu_preintegration import rotation_body_to_ned


def _csv_rows(path: Path, required: set[str]) -> tuple[list[dict], set[str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        r = csv.DictReader(stream)
        names = set(r.fieldnames or [])
        if not required.issubset(names):
            raise ValueError(f"Отсутствуют столбцы {sorted(required - names)} в {path}")
        rows = list(r)
    if not rows:
        raise ValueError(f"Пустой CSV: {path}")
    return rows, names


def _path(base: Path, name: str) -> Path:
    value = name.strip()
    if not value:
        raise ValueError("Пустой путь к входным данным")
    return (base / value).resolve()


def _triple(record: dict, prefix: str, unit: str) -> tuple[float, float, float]:
    return tuple(float(record[f"{prefix}_{axis}_{unit}"]) for axis in "xyz")


def _truth(path: Path) -> tuple[dict[int, TruthSample], dict[int, dict], set[str]]:
    rows, names = _csv_rows(
        path, {"timestamp_us", "vn_m_s", "ve_m_s", "vd_m_s", "source"}
    )
    optional_groups = [
        {"north_m", "east_m", "down_m"},
        {"qw", "qx", "qy", "qz"},
        {f"bg_{a}_rad_s" for a in "xyz"},
        {f"ba_{a}_m_s2" for a in "xyz"},
    ]
    for group in optional_groups:
        if names.intersection(group) and not group.issubset(names):
            raise ValueError("Не все поля дополнительной эталонной группы заполнены")
    result = {}
    raw = {}
    last = 0
    for row in rows:
        t = int(row["timestamp_us"])
        if t <= last:
            raise ValueError("Временные метки эталона должны возрастать")
        last = t
        position = (
            tuple(float(row[k]) for k in ("north_m", "east_m", "down_m"))
            if {"north_m", "east_m", "down_m"}.issubset(names) else None
        )
        quaternion = (
            tuple(float(row[k]) for k in ("qw", "qx", "qy", "qz"))
            if {"qw", "qx", "qy", "qz"}.issubset(names) else None
        )
        result[t] = TruthSample(
            t,
            (float(row["vn_m_s"]), float(row["ve_m_s"]), float(row["vd_m_s"])),
            row["source"],
            position,
            quaternion,
        )
        raw[t] = row
    return result, raw, names


def _snapshots(path: Path) -> dict[int, EskfSnapshot]:
    result = {}
    last = 0
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                t = int(row["timestamp_us"])
                if t <= last:
                    raise ValueError("Временные метки состояний должны возрастать")
                last = t
                result[t] = EskfSnapshot(
                    t, str(row["status"]),
                    tuple(row["position_ned_m"]),
                    tuple(row["velocity_ned_m_s"]),
                    tuple(row["quaternion_body_to_ned"]),
                    tuple(row["gyro_bias_rad_s"]),
                    tuple(row["accel_bias_m_s2"]),
                    tuple(tuple(x) for x in row["covariance_15x15"]),
                    float("nan"), 0, 0,
                )
            except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                raise ValueError(f"Некорректное состояние {path}:{line_no}") from exc
    if not result:
        raise ValueError(f"Нет состояний РФК: {path}")
    return result


def _linearizations(path: Path) -> list[LinearizationSample]:
    required = {"dt_s", "qw", "qx", "qy", "qz",
                "wx_rad_s", "wy_rad_s", "wz_rad_s",
                "fx_m_s2", "fy_m_s2", "fz_m_s2"}
    rows, _ = _csv_rows(path, required)
    sequence = []
    for row in rows:
        q = (float(row["qw"]), float(row["qx"]),
             float(row["qy"]), float(row["qz"]))
        rotation = rotation_body_to_ned(q)
        sequence.append(LinearizationSample(
            float(row["dt_s"]),
            tuple(tuple(float(v) for v in line) for line in rotation),
            tuple(float(row[f"w{a}_rad_s"]) for a in "xyz"),
            tuple(float(row[f"f{a}_m_s2"]) for a in "xyz"),
        ))
    return sequence


def _nis(path: Path) -> dict:
    rows, _ = _csv_rows(path, {"status", "innovation_nis"})
    tested = []
    statuses = Counter()
    for row in rows:
        status = row["status"]
        statuses[status] += 1
        if status in ("VISUAL_CORRECTED", "VISUAL_OUTLIER_REJECTED"):
            value = float(row["innovation_nis"])
            if not np.isfinite(value) or value < 0:
                raise ValueError("Нет NIS у обработанного визуального измерения")
            tested.append(value)
    return {
        "camera_rows": len(rows),
        "pre_gate_innovations": len(tested),
        "mean_nis_2d": float(np.mean(tested)) if tested else None,
        "fraction_above_chi2_2_99": (
            float(np.mean(np.asarray(tested) > CHI2_2_99)) if tested else None
        ),
        "status_counts": dict(sorted(statuses.items())),
        "post_gate_only_statistics_forbidden": True,
    }


def run_consistency(
    manifest: str | Path,
    output_dir: str | Path,
) -> dict:
    manifest_path = Path(manifest)
    base = manifest_path.parent
    flights, headers = _csv_rows(
        manifest_path, {"flight_id", "states_jsonl", "truth_csv"}
    )
    if len({x["flight_id"] for x in flights}) != len(flights):
        raise ValueError("Номер эталонного полета должен быть уникален")

    rows_out: list[dict] = []
    grouped: dict[str, list[dict]] = {}
    descriptions: dict[str, dict] = {}
    for record in flights:
        flight_id = record["flight_id"].strip()
        if not flight_id:
            raise ValueError("Отсутствует идентификатор полета")
        states = _snapshots(_path(base, record["states_jsonl"]))
        reference, ref_raw, ref_columns = _truth(_path(base, record["truth_csv"]))
        matched = sorted(set(states) & set(reference))
        if not matched:
            raise ValueError(f"{flight_id}: нет ни одной пары одинаковых временных меток")
        pos_aligned = record.get("origin_alignment_verified", "").strip() == "1"
        yaw_aligned = record.get("heading_alignment_verified", "").strip() == "1"
        both = pos_aligned and yaw_aligned
        bias_keys = {f"bg_{a}_rad_s" for a in "xyz"} | {
            f"ba_{a}_m_s2" for a in "xyz"
        }
        full_possible = both and bias_keys.issubset(ref_columns) and (
            reference[matched[0]].position_ned_m is not None and
            reference[matched[0]].quaternion_body_to_ned is not None
        )
        matched_rows = []
        attitude_errors = []
        complete_nees = []
        for t in matched:
            sample = nees_horizontal_velocity(states[t], reference[t])
            sample["flight_id"] = flight_id
            if reference[t].quaternion_body_to_ned is not None:
                angle = quaternion_angular_error_rad(
                    states[t].quaternion_body_to_ned,
                    reference[t].quaternion_body_to_ned,
                )
                sample["orientation_error_rad"] = angle
                attitude_errors.append(angle)
            else:
                sample["orientation_error_rad"] = None
            if full_possible:
                row = ref_raw[t]
                full = nees_full_15(
                    states[t], reference[t],
                    truth_gyro_bias_rad_s=_triple(row, "bg", "rad_s"),
                    truth_accel_bias_m_s2=_triple(row, "ba", "m_s2"),
                    origin_alignment_verified=pos_aligned,
                    heading_alignment_verified=yaw_aligned,
                )
                complete_nees.append(full)
                sample["nees_full_15d"] = full
            else:
                sample["nees_full_15d"] = None
            matched_rows.append(sample)
            rows_out.append(sample)
        grouped[flight_id] = matched_rows
        description = {
            "estimated_epochs": len(states),
            "reference_epochs": len(reference),
            "exact_time_matches": len(matched),
            "unmatched_estimated_epochs": len(states) - len(matched),
            "unmatched_reference_epochs": len(reference) - len(matched),
            "attitude_angle_rmse_rad": (
                float(np.sqrt(np.mean(np.square(attitude_errors))))
                if attitude_errors else None
            ),
            "full_nees_15d_count": len(complete_nees),
            "mean_full_nees_15d_conditional": (
                float(np.mean(complete_nees)) if complete_nees else None
            ),
            "full_nees_requires_verified_origin_heading_and_bias_reference": True,
        }
        nis_file = record.get("nis_csv", "").strip()
        description["pre_gate_nis"] = (
            _nis(_path(base, nis_file)) if nis_file else None
        )
        lin_file = record.get("linearization_csv", "").strip()
        description["local_observability"] = (
            observability_velocity_only(_linearizations(_path(base, lin_file)))
            if lin_file else None
        )
        descriptions[flight_id] = description

    summary = flight_level_summary(grouped)
    summary["data_reconciliation"] = descriptions
    summary["reference_provenance_user_declaration_only"] = True
    summary["flightworthiness_assessed"] = False
    summary["caution"] = (
        "Совпадение временных шкал, систем координат, независимость эталона "
        "и калибровка ковариаций должны быть подтверждены испытательной группой."
    )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "eskf_reference_errors.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows_out[0]))
        writer.writeheader()
        writer.writerows(rows_out)
    with (out / "eskf_consistency_summary.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False, allow_nan=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Научная проверка 15-мерного РФК по независимым полетным эталонам"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(run_consistency(
        args.manifest, args.output_dir
    ), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
