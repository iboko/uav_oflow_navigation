"""Эмпирическая калибровка ковариации скорости визуальной одометрии.

Модель ковариации из ``velocity_uncertainty`` строится по ЗАДАННЫМ
погрешностям. Здесь она проверяется и масштабируется по ошибкам
относительно НЕЗАВИСИМОГО эталона (RTK-ГНСС, оптическая система
захвата движения, синтетическая истина стенда):

    e_k = v_vo,k - v_ref,k,   C_cal,k = s * C_model,k + Q_floor.

Скаляр s — оценка максимального правдоподобия по ОБУЧАЮЩИМ полетам
(каждый полет — одна единица выборки с равным весом), систематическое
смещение b — среднее по полетам. ОТЛОЖЕННЫЕ полеты используются только
для проверки: средняя NEES (ожидается 2) и доля NEES <= χ²(2, 95 %).
Соседние кадры коррелированы, поэтому решение принимается по полетам,
а не по числу кадров. Это не сертификация датчика.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from math import isfinite, sqrt
from pathlib import Path
from typing import Sequence

import numpy as np

CHI2_2_95 = 5.991464547107979


@dataclass(frozen=True)
class VelocityErrorFlight:
    name: str
    # Rows: [ex, ey] = VO - reference, m/s (same axes as the model covariance).
    errors_m_s: tuple[tuple[float, float], ...]
    # Rows: model covariance [[cxx, cxy], [cxy, cyy]] for each error row, m²/s².
    model_covariances: tuple[tuple[tuple[float, float], tuple[float, float]], ...]


@dataclass(frozen=True)
class VelocityCovarianceCalibration:
    scale: float
    bias_m_s: tuple[float, float]
    floor_variance_m2_s2: float
    fit_flights: tuple[str, ...]
    holdout_flights: tuple[str, ...]
    fit_mean_nees_2d: float
    holdout_mean_nees_2d: float
    holdout_coverage_95pct: float
    holdout_flight_details: dict
    consistent_on_holdout: bool
    independently_validated_for_flight: bool = False
    min_required_coverage: float = 0.95

    def covariance(self, model: np.ndarray) -> np.ndarray:
        """Калиброванная ковариация для одной модельной матрицы."""
        c = self.scale * np.asarray(model, dtype=float) + np.eye(2) * self.floor_variance_m2_s2
        return 0.5 * (c + c.T)

    def to_json(self) -> dict:
        return {
            "scale": self.scale,
            "bias_m_s": list(self.bias_m_s),
            "floor_variance_m2_s2": self.floor_variance_m2_s2,
            "fit_flights": list(self.fit_flights),
            "holdout_flights": list(self.holdout_flights),
            "fit_mean_nees_2d": self.fit_mean_nees_2d,
            "holdout_mean_nees_2d": self.holdout_mean_nees_2d,
            "holdout_coverage_95pct": self.holdout_coverage_95pct,
            "min_required_coverage": self.min_required_coverage,
            "holdout_flight_details": self.holdout_flight_details,
            "consistent_on_holdout": self.consistent_on_holdout,
            "independently_validated_for_flight": self.independently_validated_for_flight,
        }


def _arrays(flight: VelocityErrorFlight, min_samples: int) -> tuple[np.ndarray, np.ndarray]:
    if not flight.name.strip():
        raise ValueError("Пустой идентификатор полета")
    e = np.asarray(flight.errors_m_s, dtype=float)
    c = np.asarray(flight.model_covariances, dtype=float)
    if e.ndim != 2 or e.shape[1] != 2 or c.shape != (len(e), 2, 2):
        raise ValueError("Ошибки и модельные ковариации должны совпадать по числу")
    if not np.isfinite(e).all() or not np.isfinite(c).all():
        raise ValueError("Неконечные ошибки или ковариации")
    if len(e) < min_samples:
        raise ValueError(f"Недостаточно измерений в полете {flight.name}")
    if not np.allclose(c, np.transpose(c, (0, 2, 1)), atol=1e-12):
        raise ValueError("Несимметричная модельная ковариация")
    if np.min(np.linalg.eigvalsh(c)) <= 0:
        raise ValueError("Модельная ковариация не положительно определена")
    return e, c


def _nees(errors: np.ndarray, covariances: np.ndarray) -> np.ndarray:
    return np.einsum("ni,ni->n", errors, np.linalg.solve(covariances, errors[..., None])[..., 0])


def calibrate_velocity_covariance(
    fitting_flights: Sequence[VelocityErrorFlight],
    held_out_flights: Sequence[VelocityErrorFlight],
    *,
    min_fitting_flights: int = 3,
    min_holdout_flights: int = 2,
    min_samples_per_flight: int = 10,
    floor_variance_m2_s2: float = 0.0,
    nees_band: tuple[float, float] = (1.4, 2.8),
    min_coverage: float | None = None,
) -> VelocityCovarianceCalibration:
    """Масштаб s и смещение b по обучающим полетам, проверка на отложенных.

    MLE скалярного масштаба для гауссовой модели: s = mean(NEES)/2 по
    центрированным ошибкам. Полеты имеют равный вес. Без искусственного
    уменьшения порога: при выходе NEES из полосы или покрытия ниже
    допустимого результат помечается consistent_on_holdout = False.
    Нижняя граница покрытия по умолчанию — биномиальный допуск 3σ около
    0,95 для фактического числа отложенных отсчетов (не ниже 0,85);
    избыточная консервативность выявляется полосой средней NEES.
    """
    if not 2 <= min_fitting_flights <= 100 or not 1 <= min_holdout_flights <= 100:
        raise ValueError("Недопустимое требуемое число полетов")
    if not isfinite(floor_variance_m2_s2) or floor_variance_m2_s2 < 0:
        raise ValueError("Недопустимая минимальная дисперсия")
    fit, holdout = list(fitting_flights), list(held_out_flights)
    if len(fit) < min_fitting_flights or len(holdout) < min_holdout_flights:
        raise ValueError("Недостаточно независимых полетов для разделения данных")
    names = [f.name for f in fit + holdout]
    if len(set(names)) != len(names):
        raise ValueError("Полеты настройки и проверки пересекаются")
    train = [_arrays(f, min_samples_per_flight) for f in fit]
    test = [_arrays(f, min_samples_per_flight) for f in holdout]
    bias = np.mean([np.mean(e, axis=0) for e, _ in train], axis=0)
    per_flight_scale = []
    for e, c in train:
        floor = np.eye(2) * floor_variance_m2_s2
        per_flight_scale.append(float(np.mean(_nees(e - bias, c + floor))) / 2.)
    scale = float(np.mean(per_flight_scale))
    if not isfinite(scale) or scale <= 0:
        raise ValueError("Недопустимый масштаб ковариации")

    def calibrated(c: np.ndarray) -> np.ndarray:
        return scale * c + np.eye(2) * floor_variance_m2_s2

    fit_mean = float(np.mean([np.mean(_nees(e - bias, calibrated(c))) for e, c in train]))
    details = {}
    means, coverages = [], []
    for flight, (e, c) in zip(holdout, test):
        nees = _nees(e - bias, calibrated(c))
        mean = float(np.mean(nees))
        coverage = float(np.mean(nees <= CHI2_2_95))
        means.append(mean)
        coverages.append(coverage)
        details[flight.name] = {
            "samples": int(len(e)),
            "mean_nees_2d": mean,
            "fraction_within_chi2_2_95": coverage,
            "bias_corrected_rmse_m_s": sqrt(float(np.mean(np.sum((e - bias) ** 2, axis=1)))),
        }
    holdout_mean = float(np.mean(means))
    holdout_coverage = float(np.mean(coverages))
    samples = sum(len(e) for e, _ in test)
    if min_coverage is None:
        min_coverage = max(.85, .95 - 3. * sqrt(.95 * .05 / samples))
    consistent = (nees_band[0] <= holdout_mean <= nees_band[1] and
                  holdout_coverage >= min_coverage)
    return VelocityCovarianceCalibration(
        scale, (float(bias[0]), float(bias[1])), float(floor_variance_m2_s2),
        tuple(f.name for f in fit), tuple(f.name for f in holdout),
        fit_mean, holdout_mean, holdout_coverage, details, bool(consistent), False,
        float(min_coverage),
    )


def _read_flight(vo_csv: Path, reference_csv: Path, *, name: str,
                 max_time_offset_us: int) -> VelocityErrorFlight:
    """VO CSV (timestamp_us, vn_m_s, ve_m_s, cov_nn, cov_ne, cov_ee) и
    эталон (timestamp_us, vn_m_s, ve_m_s) — ближайшая метка в пределах допуска."""
    with reference_csv.open(newline="", encoding="utf-8-sig") as fp:
        ref = sorted((int(r["timestamp_us"]), float(r["vn_m_s"]), float(r["ve_m_s"]))
                     for r in csv.DictReader(fp))
    if not ref:
        raise ValueError(f"Пустой эталон {reference_csv}")
    stamps = np.array([r[0] for r in ref])
    errors, covs = [], []
    with vo_csv.open(newline="", encoding="utf-8-sig") as fp:
        for row in csv.DictReader(fp):
            if row.get("accepted", "1") not in ("1", "True", "true"):
                continue
            t = int(row["timestamp_us"])
            j = int(np.argmin(np.abs(stamps - t)))
            if abs(int(stamps[j]) - t) > max_time_offset_us:
                continue
            values = [float(row[k]) for k in ("vn_m_s", "ve_m_s", "cov_nn", "cov_ne", "cov_ee")]
            if not all(isfinite(v) for v in values):
                continue
            errors.append((values[0] - ref[j][1], values[1] - ref[j][2]))
            covs.append(((values[2], values[3]), (values[3], values[4])))
    return VelocityErrorFlight(name, tuple(errors), tuple(covs))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Калибровка ковариации скорости VO по независимому эталону")
    parser.add_argument("--fit", nargs="+", required=True,
                        help="Пары NAME:VO_CSV:REFERENCE_CSV обучающих полетов")
    parser.add_argument("--holdout", nargs="+", required=True,
                        help="Пары NAME:VO_CSV:REFERENCE_CSV отложенных полетов")
    parser.add_argument("--max-time-offset-us", type=int, default=5000)
    parser.add_argument("--floor-variance", type=float, default=0.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    def load(items: list[str]) -> list[VelocityErrorFlight]:
        flights = []
        for item in items:
            name, vo, reference = item.split(":", 2)
            flights.append(_read_flight(Path(vo), Path(reference), name=name,
                                        max_time_offset_us=args.max_time_offset_us))
        return flights

    result = calibrate_velocity_covariance(
        load(args.fit), load(args.holdout), floor_variance_m2_s2=args.floor_variance)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result.to_json(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
