"""Калибровка ошибок карты и VO стенда по ОТДЕЛЬНЫМ синтетическим полетам.

Каждый полет — отдельный seed (местность, шумы ИИМ), обработанный тем же
конвейером ``run_sensor_bench`` без калибровки (карта не применяется).
Ошибки берутся относительно аналитической истины стенда:

- точка земли на оптической оси по ортофотоплану N, E ->
  ``calibrate_map_error`` (смещение, ковариация, NEES на отложенных полетах);
- курс по ортофотоплану -> смещение и дисперсия, NEES 1D;
- скорость VO в осях корпуса -> ``calibrate_velocity_covariance``
  (масштаб модельной ковариации, NEES и покрытие χ²(2)).

Полеты настройки и проверки не пересекаются, оцениваемый прогон стенда
обязан иметь третий seed (проверяется ``load_bench_calibration``).
Результат применим ТОЛЬКО к синтетическому стенду; для реального БВС
та же процедура выполняется по летным записям с независимым эталоном.
"""
from __future__ import annotations

import argparse
import csv
import json
import tempfile
from math import sqrt
from pathlib import Path
from typing import Sequence

import numpy as np

from .map_error_calibration import MapErrorFlight, calibrate_map_error
from .synthetic_sensor_bench import (
    BenchConfig, _kinematics, _wrap, generate_dataset, run_sensor_bench,
)
from .velocity_covariance_calibration import (
    VelocityErrorFlight, calibrate_velocity_covariance,
)

CHI2_1_95 = 3.841458820694124


def _flight_errors(seed: int, frames: int, scenario: str, work: Path) -> dict:
    config = BenchConfig(scenario=scenario, frames=frames, seed=seed)
    sensors = generate_dataset(work / f"seed_{seed}" / "sensors", config)
    run_sensor_bench(sensors, work / f"seed_{seed}" / "evaluation")
    with (work / f"seed_{seed}" / "evaluation" / "sensor_bench_results.csv").open(
            newline="", encoding="utf-8") as fp:
        rows = list(csv.DictReader(fp))
    map_ne, map_yaw, vo_err, vo_cov = [], [], [], []
    for row in rows:
        t = int(row["timestamp_us"])
        position, velocity, _, yaw, _ = _kinematics((t - 1_000_000) * 1e-6, scenario)
        if row["map_status"] == "GEOMETRICALLY_ACCEPTED_UNVALIDATED":
            # Truth ground point of the nadir optical axis: the bench
            # vehicle is level, so it lies directly below the vehicle.
            map_ne.append((float(row["map_ground_n_m"]) - position[0],
                           float(row["map_ground_e_m"]) - position[1]))
            map_yaw.append(_wrap(float(row["map_yaw_rad"]) - yaw))
        if row["body_vx_vo_m_s"] not in ("", "nan") and np.isfinite(float(row["body_vx_vo_m_s"])):
            tm = int(row["body_time_us"])
            _, v_mid, _, yaw_mid, _ = _kinematics((tm - 1_000_000) * 1e-6, scenario)
            c, s = np.cos(yaw_mid), np.sin(yaw_mid)
            body = (c * v_mid[0] + s * v_mid[1], -s * v_mid[0] + c * v_mid[1])
            vo_err.append((float(row["body_vx_vo_m_s"]) - body[0],
                           float(row["body_vy_vo_m_s"]) - body[1]))
            cxx, cxy, cyy = (float(row[k]) for k in ("body_cov_xx", "body_cov_xy", "body_cov_yy"))
            vo_cov.append(((cxx, cxy), (cxy, cyy)))
    return {"name": f"seed_{seed}", "frames": len(rows), "map_ne": map_ne,
            "map_yaw": map_yaw, "vo_errors": vo_err, "vo_covariances": vo_cov}


def _yaw_calibration(fit: list[dict], holdout: list[dict]) -> dict:
    train = [np.asarray(f["map_yaw"]) for f in fit]
    bias = float(np.mean([np.mean(x) for x in train]))
    variance = float(np.mean([np.mean((x - bias) ** 2) for x in train]))
    if not variance > 0:
        raise ValueError("Вырожденная дисперсия курса по карте")
    details = {}
    means, coverages = [], []
    for flight in holdout:
        e = np.asarray(flight["map_yaw"]) - bias
        nees = e ** 2 / variance
        means.append(float(np.mean(nees)))
        coverages.append(float(np.mean(nees <= CHI2_1_95)))
        details[flight["name"]] = {"samples": int(len(e)), "mean_nees_1d": means[-1],
                                   "fraction_within_chi2_1_95": coverages[-1]}
    return {"bias_rad": bias, "variance_rad2": variance, "sigma_rad": sqrt(variance),
            "holdout_mean_nees_1d": float(np.mean(means)),
            "holdout_coverage_95pct": float(np.mean(coverages)),
            "holdout_flight_details": details}


def calibrate_bench(fit_seeds: Sequence[int], holdout_seeds: Sequence[int], *,
                    frames: int = 24, scenario: str = "nominal",
                    work_dir: str | Path | None = None,
                    min_samples_per_flight: int = 15) -> dict:
    if set(fit_seeds) & set(holdout_seeds):
        raise ValueError("Seed настройки и проверки пересекаются")
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(work_dir) if work_dir else Path(tmp)
        fit = [_flight_errors(s, frames, scenario, work) for s in fit_seeds]
        holdout = [_flight_errors(s, frames, scenario, work) for s in holdout_seeds]
    def map_flight(f: dict) -> MapErrorFlight:
        return MapErrorFlight(f["name"], tuple(f["map_ne"]), f["frames"], 0)
    map_cal = calibrate_map_error(
        [map_flight(f) for f in fit], [map_flight(f) for f in holdout],
        min_fitting_flights=min(3, len(fit)), min_holdout_flights=min(2, len(holdout)),
        min_samples_per_flight=min_samples_per_flight)
    def vo_flight(f: dict) -> VelocityErrorFlight:
        return VelocityErrorFlight(f["name"], tuple(f["vo_errors"]), tuple(f["vo_covariances"]))
    vo_cal = calibrate_velocity_covariance(
        [vo_flight(f) for f in fit], [vo_flight(f) for f in holdout],
        min_fitting_flights=min(3, len(fit)), min_holdout_flights=min(2, len(holdout)))
    return {
        "kind": "SYNTHETIC_BENCH_ERROR_CALIBRATION",
        "synthetic_only": True,
        "scenario": scenario,
        "frames_per_flight": frames,
        "seeds": {"fit": list(fit_seeds), "holdout": list(holdout_seeds)},
        "map_ground_point": {
            "bias_ne_m": list(map_cal.bias_ne_m),
            "covariance_ne_m2": [list(r) for r in map_cal.covariance_ne_m2],
            "holdout_mean_nees_2d": map_cal.holdout_mean_nees_2d,
            "holdout_nees_coverage_95pct": map_cal.holdout_nees_coverage_95pct,
            "holdout_flight_details": map_cal.holdout_flight_details,
        },
        "map_yaw": _yaw_calibration(fit, holdout),
        "vo_body_velocity": vo_cal.to_json(),
        "independently_validated_for_flight": False,
        "limitations": [
            "Синтетические полеты одного аналитического сценария",
            "Кадры внутри полета коррелированы; единица выборки — полет",
            "Для реального БВС нужны летные записи НИИВК с независимым эталоном",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Калибровка ошибок карты/VO по отдельным синтетическим полетам")
    parser.add_argument("--output", required=True)
    parser.add_argument("--fit-seeds", type=int, nargs="+", default=[9101, 9102, 9103])
    parser.add_argument("--holdout-seeds", type=int, nargs="+", default=[9201, 9202])
    parser.add_argument("--frames", type=int, default=24)
    args = parser.parse_args()
    result = calibrate_bench(args.fit_seeds, args.holdout_seeds, frames=args.frames)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("seeds", "map_ground_point", "map_yaw")},
                     ensure_ascii=False, indent=2)[:3000])
    print(json.dumps(result["vo_body_velocity"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
