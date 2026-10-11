#!/usr/bin/env python3
"""Проверка камерной визуальной одометрии по PPK-эталону MUN-FRL.

Две временные части ОДНОГО полета не являются независимыми полетами.
PPK используется для эталонной скорости и (через импортер) для высоты;
на обучающей части по PPK также определяются поправки курса и масштаба.
На границе обучения и проверки исключаются пары, разделяющие кадр.
Результаты нельзя переносить в PX4 как калибровку бортового датчика.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

CHI2_2_95 = 5.991464547107979
PROFILE = {"sigma_min_m_s": 0.05, "sigma_rate_rad_s": 0.0015, "sigma_h_rel": 0.01}


def load(manifest: Path, vo: Path, *, require_ppk_fix: bool = True) -> list[dict]:
    """Согласовать только принятые пары с монотонным PPK-эталоном.

    По умолчанию допускаются только пары, у которых обе отметки PPK Q=1
    (fixed). Пропущенные и некорректные эталонные данные не подменяются.
    """
    with manifest.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"timestamp_us", "true_n_m", "true_e_m", "height_agl_m"}
        if require_ppk_fix:
            required.add("ppk_quality")
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"В manifest отсутствуют поля {sorted(required - set(reader.fieldnames or []))}")
        man = list(reader)
    if len(man) < 3:
        raise ValueError("Недостаточно кадров с эталоном")
    times = [int(r["timestamp_us"]) for r in man]
    if any(a >= b for a, b in zip(times, times[1:])):
        raise ValueError("Метки manifest не возрастают строго")
    index = {str(t): i for i, t in enumerate(times)}
    rows: list[dict] = []
    seen: set[str] = set()
    with vo.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required_vo = {"timestamp_us", "accepted", "vn_m_s", "ve_m_s"}
        if not required_vo.issubset(reader.fieldnames or []):
            raise ValueError("В VO отсутствуют обязательные поля")
        for r in reader:
            if r["accepted"] not in ("1", "true", "True"):
                continue
            stamp = r["timestamp_us"]
            if stamp in seen:
                raise ValueError(f"Дублирующийся принятый кадр VO: {stamp}")
            seen.add(stamp)
            if stamp not in index:
                raise ValueError(f"Нет эталона PPK для кадра VO {stamp}")
            i = index[stamp]
            if i == 0:
                raise ValueError("Первый кадр не может содержать принятую межкадровую скорость")
            a, b = man[i - 1], man[i]
            if require_ppk_fix and any(str(p["ppk_quality"]).strip() != "1" for p in (a, b)):
                continue
            dt = (times[i] - times[i - 1]) * 1e-6
            if not math.isfinite(dt) or dt <= 0:
                raise ValueError("Недопустимый интервал пары кадров")
            na, ea, nb, eb = (float(a["true_n_m"]), float(a["true_e_m"]),
                              float(b["true_n_m"]), float(b["true_e_m"]))
            h = 0.5 * (float(a["height_agl_m"]) + float(b["height_agl_m"]))
            velocity = np.array([float(r["vn_m_s"]), float(r["ve_m_s"])])
            ref = np.array([(nb - na) / dt, (eb - ea) / dt])
            if not (np.isfinite(velocity).all() and np.isfinite(ref).all()
                    and math.isfinite(h) and h > 0):
                raise ValueError(f"Неконечные значения или недопустимая высота: {stamp}")
            rows.append({
                "t": (times[i] - times[0]) * 1e-6,
                "start_t": (times[i - 1] - times[0]) * 1e-6,
                "start_stamp_us": times[i - 1],
                "end_stamp_us": times[i],
                "vo": velocity,
                "ref": ref,
                "h": h,
                "dt": dt,
            })
    if not rows:
        raise ValueError("Нет принятых пар VO с допустимым PPK-эталоном")
    return rows


def rot(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s], [s, c]])


def sigma2(p: dict, h: float, v: float) -> float:
    return p["sigma_min_m_s"] ** 2 + (h * p["sigma_rate_rad_s"]) ** 2 + (v * p["sigma_h_rel"]) ** 2


def metrics(rows: list[dict], heading_deg: float, scale: float, params: dict | None) -> dict:
    if not rows:
        return {"intervals": 0}
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Масштаб скорости должен быть положительным")
    e = np.array([rot(-heading_deg) @ r["vo"] / scale - r["ref"] for r in rows])
    out = {
        "intervals": len(rows),
        "rmse_m_s": float(np.sqrt(np.mean(np.sum(e * e, axis=1)))),
        "median_abs_m_s": float(np.median(np.linalg.norm(e, axis=1))),
        "mean_error_ne_m_s": e.mean(axis=0).round(3).tolist(),
        "mean_height_m": float(np.mean([r["h"] for r in rows])),
        "mean_speed_m_s": float(np.mean([np.linalg.norm(r["ref"]) for r in rows])),
    }
    if params is not None:
        variances = np.array([sigma2(params, r["h"], float(np.linalg.norm(r["ref"])))
                              for r in rows])
        if not np.isfinite(variances).all() or np.any(variances <= 0):
            raise ValueError("В модели ковариации имеются нулевые или отрицательные дисперсии")
        nees = np.sum(e * e, axis=1) / variances
        out["mean_nees"] = float(nees.mean())
        out["nees_le_chi2_95_fraction"] = float(np.mean(nees <= CHI2_2_95))
    return out


def fit_nonnegative_variances(matrix: np.ndarray, squared_errors: np.ndarray) -> np.ndarray:
    """Точный перебор активных подмножеств для малой NNLS-задачи (3 столбца).

    Прежний алгоритм необратимо исключал наиболее отрицательный столбец;
    такой шаг не гарантирует минимума по всем неотрицательным коэффициентам.
    """
    if matrix.ndim != 2 or matrix.shape[1] != 3 or squared_errors.shape != (len(matrix),):
        raise ValueError("Неверная размерность NNLS-задачи")
    best = None
    best_loss = math.inf
    for mask in range(1 << 3):
        active = [k for k in range(3) if mask & (1 << k)]
        coef = np.zeros(3)
        if active:
            sol, *_ = np.linalg.lstsq(matrix[:, active], squared_errors, rcond=None)
            if np.min(sol) < -1e-12:
                continue
            coef[active] = np.maximum(sol, 0)
        loss = float(np.sum((matrix @ coef - squared_errors) ** 2))
        if loss < best_loss:
            best, best_loss = coef, loss
    if best is None or not np.isfinite(best).all():
        raise ValueError("Не удалось оценить дисперсии")
    return best


def evaluate(rows: list[dict], split_s: float, holdout_gap_s: float = 0.0) -> dict:
    if not math.isfinite(split_s) or split_s <= 0:
        raise ValueError("Граница обучения должна быть положительной")
    if not math.isfinite(holdout_gap_s) or holdout_gap_s < 0:
        raise ValueError("Временной зазор проверки не может быть отрицательным")

    cal = [r for r in rows if r["t"] < split_s]
    # Валидируется только пара, НАЧАВШАЯСЯ после границы. Ни один кадр
    # не может одновременно участвовать в обучении и проверке.
    val = [r for r in rows if r["start_t"] >= split_s + holdout_gap_s - 1e-9]
    if len(cal) < 5 or len(val) < 5:
        raise ValueError("Недостаточно независимых по границе интервалов для обучения/проверки")
    cal_frames = {stamp for r in cal for stamp in (r["start_stamp_us"], r["end_stamp_us"])}
    val_frames = {stamp for r in val for stamp in (r["start_stamp_us"], r["end_stamp_us"])}
    if cal_frames & val_frames:
        raise ValueError("Утечка данных: общий кадр в обучении и проверке")

    fast = [r for r in cal if np.linalg.norm(r["ref"]) > 2.0]
    if len(fast) < 5:
        raise ValueError("Недостаточно быстрых интервалов для курсовой выставки")
    diffs = [math.atan2(r["vo"][1], r["vo"][0]) -
             math.atan2(r["ref"][1], r["ref"][0]) for r in fast]
    heading = math.degrees(math.atan2(np.median(np.sin(diffs)), np.median(np.cos(diffs))))
    scale = float(np.median([np.linalg.norm(r["vo"]) / np.linalg.norm(r["ref"]) for r in fast]))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Невозможно определить масштаб скорости")

    e2 = np.array([float(np.sum((rot(-heading) @ r["vo"] / scale - r["ref"]) ** 2)) / 2.
                   for r in cal])
    matrix = np.array([[1., r["h"] ** 2, float(np.linalg.norm(r["ref"])) ** 2]
                       for r in cal])
    coef = fit_nonnegative_variances(matrix, e2)
    fitted = {
        "sigma_min_m_s": math.sqrt(coef[0]),
        "sigma_rate_rad_s": math.sqrt(coef[1]),
        "sigma_h_rel": math.sqrt(coef[2]),
    }

    return {
        "type": "MUNFRL_REAL_DATA_VO_VELOCITY_CHECK",
        "reference": "PPK (RTKLIB) positions of the same frame pair",
        "validation_scope": "SINGLE_FLIGHT_TEMPORAL_SPLIT_NOT_INDEPENDENT_FLIGHTS",
        "independent_flight_validation": False,
        "ppk_used_for_height": True,
        "ppk_used_for_heading_and_scale_fit": True,
        "covariance_applied_to_px4": False,
        "boundary_frames_shared": 0,
        "split_s": split_s,
        "holdout_gap_s": holdout_gap_s,
        "excluded_boundary_intervals": len(rows) - len(cal) - len(val),
        "pair_interval_s": float(np.median([r["dt"] for r in rows])),
        "calibration": {
            "heading_offset_deg": heading,
            "speed_scale_vo_over_ref": scale,
            "fitted_noise_model": fitted,
            "raw_before_heading_and_scale": metrics(cal, 0.0, 1.0, None),
            "after": metrics(cal, heading, scale, fitted),
        },
        "validation": {
            "profile_constants": PROFILE,
            "with_profile_constants": metrics(val, heading, scale, PROFILE),
            "with_fitted_model": metrics(val, heading, scale, fitted),
            "without_scale_correction": metrics(val, heading, 1.0, fitted),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("manifest", type=Path)
    ap.add_argument("vo_csv", type=Path)
    ap.add_argument("--split-s", type=float, default=12.0)
    ap.add_argument("--holdout-gap-s", type=float, default=0.0,
                    help="Дополнительный временной зазор между обучением и проверкой")
    ap.add_argument("--allow-nonfixed-ppk", action="store_true",
                    help="Только диагностический запуск: разрешить пары с PPK Q != 1")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    rows = load(args.manifest, args.vo_csv, require_ppk_fix=not args.allow_nonfixed_ppk)
    report = evaluate(rows, args.split_s, args.holdout_gap_s)
    report["require_ppk_fix"] = not args.allow_nonfixed_ppk
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
