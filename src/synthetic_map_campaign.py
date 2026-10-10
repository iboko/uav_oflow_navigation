"""Reproducible *synthetic*, statistically defined map/INS stress campaign.

This is a 2-D error surrogate, NOT an image renderer, PX4 SITL flight, or
independent real-world sensor validation. True trajectory is created from
analytic functions; simulated map and inertial positions have controlled
shared errors, drift, delayed samples, outages and false map associations.
Truth is used ONLY for training-set calibration and final scoring, never for
holdout candidate selection or integrity gating.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .map_covariance_intersection import covariance_intersection_2d
from .map_error_calibration import MapErrorFlight, calibrate_map_error


@dataclass(frozen=True)
class CampaignConfig:
    seed: int = 14921
    fit_flights: int = 3
    holdout_flights: int = 3
    frames_per_flight: int = 120
    dt_s: float = 0.1
    max_map_prior_disagreement_m: float = 6.0
    false_fix_m: float = 6.0
    map_outlier_shift_m: float = 25.0

    def validate(self) -> None:
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("Случайный seed должен быть неотрицательным целым")
        if not (3 <= self.fit_flights <= 30 and 2 <= self.holdout_flights <= 30
                and 40 <= self.frames_per_flight <= 2000):
            raise ValueError("Не хватает независимых прогонов или отсчетов")
        if not (0.01 <= self.dt_s <= 0.5):
            raise ValueError("Неверный шаг модельного времени")
        if not (0 < self.max_map_prior_disagreement_m <= 50
                and 0 < self.false_fix_m <= 100
                and 2 < self.map_outlier_shift_m <= 1000):
            raise ValueError("Неверные пороги и модельный сдвиг ложной привязки")


_SCENARIOS = ("nominal", "shared_correlation", "false_match_burst", "camera_outage")


def _trajectory(t: np.ndarray) -> np.ndarray:
    """Known synthetic N/E reference; no hidden georeferencing."""
    return np.column_stack((
        1.0 * t + 1.3 * np.sin(0.19 * t),
        0.35 * t + 1.7 * np.sin(0.13 * t)
    ))


def _colored_noise(rng: np.random.Generator, n: int, scale: float,
                   persistence: float) -> np.ndarray:
    state = np.zeros(2)
    data = np.zeros((n, 2))
    for i in range(n):
        state = persistence * state + np.sqrt(1 - persistence**2) * (
            scale * rng.normal(size=2)
        )
        data[i] = state
    return data


def _flight(rng: np.random.Generator, cfg: CampaignConfig,
            scenario: str, *, fitting: bool) -> dict:
    n = cfg.frames_per_flight
    t = np.arange(n) * cfg.dt_s
    truth = _trajectory(t)
    shared_scale = 1.0 if scenario in (
        "shared_correlation", "false_match_burst") else 0.35
    shared = _colored_noise(rng, n, shared_scale, 0.88)
    map_ind = _colored_noise(rng, n, 0.55, 0.60)
    prior_ind = _colored_noise(rng, n, 0.7, 0.8)
    map_bias = np.array((0.7, -0.3))
    map_error = map_bias + 0.65 * shared + map_ind
    # Include small drift not explicitly corrected by this map-only campaign.
    prior_error = 0.75 * shared + prior_ind + 0.035 * t[:, None] * np.array((1., -.3))
    map_position = truth + map_error
    prior_position = truth + prior_error

    available = rng.uniform(size=n) > 0.08
    bad_map = np.zeros(n, dtype=bool)
    if not fitting and scenario == "false_match_burst":
        start = n // 3
        bad_map[start:start + min(5, n // 10)] = True
        map_position[bad_map] += np.array((cfg.map_outlier_shift_m, -5.0))
    if not fitting and scenario == "camera_outage":
        available[n//3:2*n//3] = False

    # Fixed camera-arrival delay is synthetic metadata. It does not alter the
    # analytic truth or pretend to test raw IMU propagation.
    delay_us = rng.integers(12_000, 190_001, size=n)
    timestamp_us = (1_000_000 + np.round(t*1e6)).astype(np.int64)
    arrival_us = timestamp_us + delay_us
    # No truth-dependent acceptance during holdout.
    return {
        "truth": truth, "map": map_position, "prior": prior_position,
        "available": available, "bad_map": bad_map,
        "timestamp_us": timestamp_us, "arrival_us": arrival_us,
    }


def _cov_estimate_train(flights: list[dict], field: str) -> tuple[np.ndarray, np.ndarray]:
    arrays = [
        (f[field] - f["truth"])[f["available"]]
        for f in flights
    ]
    bias = np.mean([np.mean(x, axis=0) for x in arrays], axis=0)
    cov = np.mean([
        (x-bias).T@(x-bias)/len(x) for x in arrays
    ], axis=0)
    if np.linalg.eigvalsh(cov)[0] <= 0:
        raise ValueError("Вырожденная учебная ковариация прогноза")
    return bias, cov


def _metrics(errors: list[np.ndarray]) -> dict:
    if not errors:
        return {"evaluated_frames":0,"rmse_m":None,"p95_m":None}
    arr = np.asarray(errors, dtype=float)
    radius = np.linalg.norm(arr, axis=1)
    return {
        "evaluated_frames": len(radius),
        "rmse_m": float(np.sqrt(np.mean(radius**2))),
        "p95_m": float(np.percentile(radius, 95)),
    }


def run_campaign(config: CampaignConfig = CampaignConfig()) -> dict:
    config.validate()
    scenarios = {}
    for scenario_number, scenario in enumerate(_SCENARIOS):
        rng = np.random.default_rng(config.seed + scenario_number * 100003)
        train = [
            _flight(rng, config, scenario, fitting=True)
            for _ in range(config.fit_flights)
        ]
        holdout = [
            _flight(rng, config, scenario, fitting=False)
            for _ in range(config.holdout_flights)
        ]
        training_errors = [
            MapErrorFlight(
                f"T{i}", tuple(map(tuple, (f["map"]-f["truth"])[f["available"]])),
                len(f["available"]), 0
            ) for i, f in enumerate(train)
        ]
        # Truth enters calibration only through the fitting flights.
        heldout_errors = []
        for i, flight in enumerate(holdout):
            residual = (flight["map"]-flight["truth"])[flight["available"]]
            false = int(np.sum(np.linalg.norm(residual, axis=1) > config.false_fix_m))
            heldout_errors.append(MapErrorFlight(
                f"H{i}", tuple(map(tuple, residual)),
                len(flight["available"]), false
            ))
        calibration = calibrate_map_error(
            training_errors, heldout_errors, min_samples_per_flight=30
        )
        bias_map = np.asarray(calibration.bias_ne_m)
        cov_map = np.asarray(calibration.covariance_ne_m2)
        prior_bias, cov_prior = _cov_estimate_train(train, "prior")

        prior_errors: list[np.ndarray] = []
        mapped_errors: list[np.ndarray] = []
        ci_errors: list[np.ndarray] = []
        total_false_available = 0
        false_map_gate_passed = 0
        false_ci_bad = 0
        map_disagreement_rejected = 0
        missed = 0
        details = []
        for flight_index, flight in enumerate(holdout):
            per_flight = {
                "flight_id":f"H{flight_index}",
                "candidate_frames":int(np.sum(flight["available"])),
                "ci_accepted":0,
                "ci_rejected":0,
                "ci_error_m":[],
            }
            for index in range(config.frames_per_flight):
                if not flight["available"][index]:
                    missed += 1
                    continue
                truth = flight["truth"][index]
                prior = flight["prior"][index] - prior_bias
                measured = flight["map"][index] - bias_map
                prior_errors.append(prior-truth)
                mapped_errors.append(measured-truth)
                actually_false = np.linalg.norm(flight["map"][index]-truth) > config.false_fix_m
                total_false_available += bool(actually_false)
                # The gate sees only available predictions, not reference truth.
                if np.linalg.norm(prior-measured) > config.max_map_prior_disagreement_m:
                    map_disagreement_rejected += 1
                    per_flight["ci_rejected"] += 1
                    continue
                estimate, p_ci, w = covariance_intersection_2d(
                    prior, cov_prior, measured, cov_map, grid_steps=61
                )
                if not np.isfinite(estimate).all() or np.linalg.eigvalsh(p_ci)[0] <= 0:
                    raise ValueError("Численный отказ CI")
                err = estimate-truth
                ci_errors.append(err)
                per_flight["ci_error_m"].append(float(np.linalg.norm(err)))
                per_flight["ci_accepted"] += 1
                false_map_gate_passed += bool(actually_false)
                false_ci_bad += bool(np.linalg.norm(err) > config.false_fix_m)
            per_flight["ci_rmse_m"] = (
                float(np.sqrt(np.mean(np.square(per_flight["ci_error_m"]))))
                if per_flight["ci_error_m"] else None
            )
            del per_flight["ci_error_m"]
            details.append(per_flight)

        scenarios[scenario] = {
            "prior":_metrics(prior_errors),
            "map_bias_corrected":_metrics(mapped_errors),
            "covariance_intersection_on_gated_frames":_metrics(ci_errors),
            "total_holdout_frames":config.holdout_flights*config.frames_per_flight,
            "camera_unavailable_frames":missed,
            "false_map_observations_with_truth":total_false_available,
            "false_map_observations_passing_difference_gate":false_map_gate_passed,
            "ci_position_error_exceeding_false_fix_threshold":false_ci_bad,
            "map_prior_disagreement_rejections":map_disagreement_rejected,
            "covariance_source":"SYNTHETIC_TRAINING_NOT_INDEPENDENTLY_VALIDATED",
            "calibration_train_flights":list(calibration.fit_flights),
            "calibration_holdout_flights":list(calibration.holdout_flights),
            "simulated_arrival_delay_range_us":[
                min(int(np.min(f["arrival_us"]-f["timestamp_us"])) for f in holdout),
                max(int(np.max(f["arrival_us"]-f["timestamp_us"])) for f in holdout),
            ],
            "per_flight":details,
        }
    return {
        "campaign_type":"SYNTHETIC_STATISTICAL_SURROGATE_ONLY",
        "config":asdict(config),
        "scenarios":scenarios,
        "passed_flight_validation":False,
        "px4_eskf_state_mutated":False,
        "warnings":[
            "Эталон аналитический, измерений НИИВК нет и не предполагается",
            "Это статистическая модель 2D позиций, не симулятор реальной оптики, рельефа или ИИМ",
            "Данные отдельных кадров автокоррелированы; метрики сценариев не являются летными нормативами",
            "Задержки камер сгенерированы и записаны, без моделирования перепропагации 15D РФК",
            "Выход CI не передается в PX4 и не меняет 15-мерный ESKF",
            "Погрешности, подобранные на синтетике, нельзя использовать как подтвержденные ковариации БВС",
        ],
    }


def main() -> None:
    parser=argparse.ArgumentParser(
        description="Воспроизводимая синтетическая проверка ошибки карты и CI без записей НИИВК"
    )
    parser.add_argument("--output-dir",required=True)
    parser.add_argument("--seed",type=int,default=14921)
    parser.add_argument("--frames-per-flight",type=int,default=120)
    args=parser.parse_args()
    report=run_campaign(CampaignConfig(
        seed=args.seed,frames_per_flight=args.frames_per_flight
    ))
    out=Path(args.output_dir)
    out.mkdir(parents=True,exist_ok=True)
    with (out/"synthetic_map_campaign.json").open("w",encoding="utf-8") as f:
        json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
    with (out/"synthetic_map_scenarios.csv").open(
        "w",newline="",encoding="utf-8"
    ) as f:
        fields=["scenario","prior_rmse_m","map_rmse_m","ci_rmse_m",
                "holdout_frames","camera_unavailable_frames",
                "false_map_observations","false_maps_passing_gate",
                "ci_errors_exceeding_threshold"]
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        for name,s in report["scenarios"].items():
            w.writerow({
                "scenario":name,
                "prior_rmse_m":s["prior"]["rmse_m"],
                "map_rmse_m":s["map_bias_corrected"]["rmse_m"],
                "ci_rmse_m":s["covariance_intersection_on_gated_frames"]["rmse_m"],
                "holdout_frames":s["total_holdout_frames"],
                "camera_unavailable_frames":s["camera_unavailable_frames"],
                "false_map_observations":s["false_map_observations_with_truth"],
                "false_maps_passing_gate":s["false_map_observations_passing_difference_gate"],
                "ci_errors_exceeding_threshold":s["ci_position_error_exceeding_false_fix_threshold"],
            })
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
