#!/usr/bin/env python3
"""Калибровка модели СКО скорости VO по НЕСКОЛЬКИМ полетам MUN-FRL.

Вход — пары (manifest импортера, visual_odometry.csv) для каждого полета.
Единица выборки — полет (соседние интервалы коррелированы). Проверка —
перекрестная с исключением одного полета (leave-one-flight-out):

  для каждого полета k
    1. модель  σv² = σmin² + (h·σrate)² + (|v|·σh,rel)²  — NNLS по остальным
       полетам с равным весом полета (тот же перебор активных множеств, что
       в munfrl_vo_eval.fit_nonnegative_variances) и проверка различимости;
    2. масштаб s и смещение b — src.velocity_covariance_calibration
       по тем же обучающим полетам;
    3. полет k — только проверка: средняя NEES (ожидается 2) и доля
       NEES ≤ χ²(2; 95 %); для сравнения — константы профиля `camera`.

Курс Xsens в MUN-FRL не привязан к северу, а высота над землей берется
по PPK при допущении плоской земли. Поэтому для каждого полета по его
быстрым интервалам оцениваются сдвиг курса и масштаб скорости
(в системе их дает карта/магнитометр и дальномер). Это единственное
использование эталона полета проверки до расчета его NEES; оно
указывается в отчете.

Требование src.velocity_covariance_calibration по умолчанию — 3 обучающих
и 2 отложенных полета; при меньшем числе полетов результат помечается
как не удовлетворяющий этому критерию, а не выдается за калибровку.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.velocity_covariance_calibration import (  # noqa: E402
    CHI2_2_95, VelocityErrorFlight, calibrate_velocity_covariance)
from munfrl_vo_eval import PROFILE, fit_nonnegative_variances, load, rot, sigma2  # noqa: E402

REQUIRED_FIT, REQUIRED_HOLDOUT = 3, 2
MIN_FAST_INTERVALS = 5


HEIGHT_MODELS = ("assumed", "effective")


def align_flight(name: str, rows: list[dict], *, correct_scale: bool = True,
                 height_model: str = "assumed") -> dict:
    """Сдвиг курса и масштаб по быстрым интервалам, ошибки в осях NE."""
    fast = [r for r in rows if np.linalg.norm(r["ref"]) > 2.0]
    if len(fast) < MIN_FAST_INTERVALS:
        raise ValueError(f"{name}: недостаточно интервалов быстрее 2 м/с для выставки курса")
    # Least-squares similarity (Procrustes) vo ≈ s·R(θ)·ref: unbiased under
    # zero-mean VO noise, unlike a median of |vo|/|ref| (E|ref+n| > |ref|).
    ref = np.array([r["ref"] for r in fast])
    vo = np.array([r["vo"] for r in fast])
    cross = float(np.sum(ref[:, 0] * vo[:, 1] - ref[:, 1] * vo[:, 0]))
    dot = float(np.sum(ref[:, 0] * vo[:, 0] + ref[:, 1] * vo[:, 1]))
    heading = math.degrees(math.atan2(cross, dot))
    scale = math.hypot(cross, dot) / float(np.sum(ref * ref))
    if not correct_scale:
        scale = 1.0
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"{name}: недопустимый масштаб скорости")
    errors = np.array([rot(-heading) @ r["vo"] / scale - r["ref"] for r in rows])
    # Height in the noise model:
    #  "assumed"   — the height VO itself used (PPK above the takeoff point);
    #  "effective" — h/scale, i.e. the scale error attributed to terrain
    #                height only. The choice is made by cross-validation.
    if height_model not in HEIGHT_MODELS:
        raise ValueError(f"Неизвестная модель высоты {height_model}")
    h_eff = np.array([r["h"] / scale for r in rows])
    h_model = h_eff if height_model == "effective" else np.array([r["h"] for r in rows])
    features = np.array([[1., h ** 2, float(np.linalg.norm(r["ref"])) ** 2]
                         for h, r in zip(h_model, rows)])
    return {
        "name": name,
        "heading_offset_deg": heading,
        "speed_scale": scale,
        "errors": errors,
        "features": features,
        "intervals": len(rows),
        "fast_intervals": len(fast),
        "mean_height_m": float(np.mean([r["h"] for r in rows])),
        "height_range_m": [float(min(r["h"] for r in rows)), float(max(r["h"] for r in rows))],
        "effective_height_range_m": [float(h_eff.min()), float(h_eff.max())],
        "mean_speed_m_s": float(np.mean([np.linalg.norm(r["ref"]) for r in rows])),
        "pair_interval_s": float(np.median([r["dt"] for r in rows])),
    }


def fit_noise_model(flights: list[dict]) -> dict:
    """NNLS по |e|²/2 с равным весом каждого полета и проверкой различимости."""
    blocks, targets = [], []
    for f in flights:
        w = 1.0 / math.sqrt(len(f["errors"]))
        blocks.append(f["features"] * w)
        targets.append(np.sum(f["errors"] ** 2, axis=1) / 2.0 * w)
    matrix, target = np.vstack(blocks), np.concatenate(targets)
    norms = np.linalg.norm(matrix, axis=0)
    singular = np.linalg.svd(matrix / np.where(norms > 0, norms, 1.0), compute_uv=False)
    rank = int(np.sum(singular > singular[0] * 1e-8))
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else math.inf
    coef = fit_nonnegative_variances(matrix, target)
    # Statistical precision, not only design conditioning: |e|²/2 is
    # χ²-distributed, so a small variance term can be unresolved although
    # the design matrix is well conditioned. Heteroscedasticity-robust
    # (sandwich) standard errors of the variance coefficients.
    residual = target - matrix @ coef
    gram_inv = np.linalg.pinv(matrix.T @ matrix)
    cov = gram_inv @ (matrix.T * residual ** 2) @ matrix @ gram_inv
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    names = ("sigma_min_m_s", "sigma_rate_rad_s", "sigma_h_rel")
    return {
        "sigma_min_m_s": math.sqrt(coef[0]),
        "sigma_rate_rad_s": math.sqrt(coef[1]),
        "sigma_h_rel": math.sqrt(coef[2]),
        "variance_coefficients": coef.tolist(),
        "variance_coefficient_se": se.tolist(),
        "resolved_terms": [n for n, c, e in zip(names, coef, se) if c > 2.0 * e],
        "design_rank": rank,
        "normalized_condition_number": condition if math.isfinite(condition) else None,
        "identifiable": bool(rank == 3 and math.isfinite(condition) and condition < 1e4),
    }


def _params(model: dict) -> dict:
    return {k: model[k] for k in ("sigma_min_m_s", "sigma_rate_rad_s", "sigma_h_rel")}


def _covariances(flight: dict, params: dict) -> np.ndarray:
    variances = np.array([sigma2(params, math.sqrt(x[1]), math.sqrt(x[2])) for x in flight["features"]])
    if np.any(~np.isfinite(variances)) or np.any(variances <= 0):
        raise ValueError("Модель дает неположительную дисперсию")
    return variances[:, None, None] * np.eye(2)[None]


def _error_flight(flight: dict, params: dict) -> VelocityErrorFlight:
    covs = _covariances(flight, params)
    return VelocityErrorFlight(flight["name"], tuple(map(tuple, flight["errors"])),
                               tuple(tuple(map(tuple, c)) for c in covs))


def nees_stats(flight: dict, params: dict, *, scale: float = 1.0,
               bias: tuple[float, float] = (0.0, 0.0)) -> dict:
    e = flight["errors"] - np.asarray(bias)
    var = scale * np.array([c[0, 0] for c in _covariances(flight, params)])
    nees = np.sum(e * e, axis=1) / var
    return {"mean_nees_2d": float(nees.mean()),
            "fraction_within_chi2_2_95": float(np.mean(nees <= CHI2_2_95))}


def leave_one_flight_out(flights: list[dict]) -> list[dict]:
    folds = []
    for k, held in enumerate(flights):
        train = [f for i, f in enumerate(flights) if i != k]
        model = fit_noise_model(train)
        params = _params(model)
        cal = calibrate_velocity_covariance(
            [_error_flight(f, params) for f in train], [_error_flight(held, params)],
            min_fitting_flights=len(train), min_holdout_flights=1,
            min_samples_per_flight=10)
        folds.append({
            "holdout_flight": held["name"],
            "fit_flights": [f["name"] for f in train],
            "fitted_noise_model": model,
            "covariance_scale": cal.scale,
            "bias_m_s": list(cal.bias_m_s),
            "holdout": cal.holdout_flight_details[held["name"]],
            "holdout_consistent": cal.consistent_on_holdout,
            "holdout_min_required_coverage": cal.min_required_coverage,
            "profile_constants_on_holdout": nees_stats(held, PROFILE),
        })
    return folds


def run(items: list[tuple[str, Path, Path]], *, correct_scale: bool = True,
        height_model: str = "assumed") -> dict:
    if len(items) < 2:
        raise ValueError("Нужно не меньше двух полетов")
    names = [n for n, _, _ in items]
    if len(set(names)) != len(names):
        raise ValueError("Имена полетов повторяются")
    flights = [align_flight(n, load(m, v), correct_scale=correct_scale, height_model=height_model)
               for n, m, v in items]
    folds = leave_one_flight_out(flights) if len(flights) >= 3 else []
    final = fit_noise_model(flights)
    all_consistent = bool(folds) and all(f["holdout_consistent"] for f in folds)
    return {
        "type": "MUNFRL_MULTI_FLIGHT_VO_COVARIANCE_CALIBRATION",
        "reference": "PPK (RTKLIB fix) positions of the same frame pair",
        "flights": [{k: v for k, v in f.items() if k not in ("errors", "features")} for f in flights],
        "per_flight_reference_use": ("heading offset and speed scale estimated per flight from its "
                                     "own PPK reference (Xsens heading is not north-referenced; "
                                     "flat-ground height)" if correct_scale else
                                     "heading offset estimated per flight from its own PPK reference"),
        "speed_scale_corrected": correct_scale,
        "height_model": height_model,
        "leave_one_flight_out": folds,
        "all_holdout_folds_consistent": all_consistent,
        "final_model_all_flights": final,
        "profile_constants": PROFILE,
        "required_flights_for_calibration": {"fit": REQUIRED_FIT, "holdout": REQUIRED_HOLDOUT},
        "meets_required_flight_count": len(flights) >= REQUIRED_FIT + REQUIRED_HOLDOUT,
        "covariance_applied_to_px4": False,
        "validated_for_flight": False,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("flights", nargs="+", help="NAME:MANIFEST_CSV:VO_CSV")
    ap.add_argument("--no-scale-correction", action="store_true")
    ap.add_argument("--height-model", choices=HEIGHT_MODELS, default="assumed",
                    help="assumed: height used by VO; effective: h/scale")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    items = []
    for spec in args.flights:
        parts = spec.split(":")
        if len(parts) != 3:
            ap.error(f"ожидается NAME:MANIFEST:VO, получено {spec!r}")
        items.append((parts[0], Path(parts[1]), Path(parts[2])))
    report = run(items, correct_scale=not args.no_scale_correction, height_model=args.height_model)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
