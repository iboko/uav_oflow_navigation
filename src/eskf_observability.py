"""Локальная наблюдаемость 15-мерного РФК для наборов коррекций.

Для номинальной траектории по ИИМ строится матрица наблюдаемости
O = [H_0; H_1 Φ(1,0); ...; H_N Φ(N,0)] с теми же якобианами, что
в ``error_state_inertial_filter`` и ``eskf15.cpp``. Столбцы
нормируются априорными СКО, ранг определяется по сингулярным числам.
Для каждой компоненты состояния выдается доля, лежащая в
ненаблюдаемом подпространстве (1 — полностью ненаблюдаема).

Это анализ линеаризованной модели на конечном окне: он показывает,
КАКИЕ состояния нельзя оценить при данном наборе датчиков, но не
оценивает скорость сходимости и не заменяет летных испытаний.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from math import cos, sin
from pathlib import Path

import numpy as np

from .error_state_inertial_filter import (
    GRAVITY, continuous_error_jacobian, exp_transition, skew,
)
from .imu_preintegration import (
    quat_exponential, quat_product, rotation_body_to_ned,
)

STATE_NAMES = (
    "p_n", "p_e", "p_d", "v_n", "v_e", "v_d",
    "theta_x", "theta_y", "theta_z",
    "bg_x", "bg_y", "bg_z", "ba_x", "ba_y", "ba_z",
)
PRIOR_SIGMA = np.array([10., 10., 10., 1., 1., 1., .1, .1, .1,
                        .02, .02, .02, .2, .2, .2])
AIDING = ("velocity_ne", "body_velocity_xy", "yaw", "position_ne")


@dataclass(frozen=True)
class Scenario:
    name: str
    motion: str  # "hover" or "maneuver"
    aiding: tuple[str, ...]


SCENARIOS = (
    Scenario("hover_velocity_ne", "hover", ("velocity_ne",)),
    Scenario("hover_body_velocity", "hover", ("body_velocity_xy",)),
    Scenario("hover_body_velocity_yaw", "hover", ("body_velocity_xy", "yaw")),
    Scenario("hover_body_velocity_yaw_map", "hover",
             ("body_velocity_xy", "yaw", "position_ne")),
    Scenario("maneuver_velocity_ne", "maneuver", ("velocity_ne",)),
    Scenario("maneuver_body_velocity", "maneuver", ("body_velocity_xy",)),
    Scenario("maneuver_body_velocity_map", "maneuver",
             ("body_velocity_xy", "position_ne")),
    Scenario("maneuver_all", "maneuver", AIDING),
)


def _imu(t: float, motion: str) -> tuple[np.ndarray, np.ndarray]:
    """Body rate and specific force (FRD) of the nominal trajectory."""
    if motion == "hover":
        return np.zeros(3), np.array([0., 0., -GRAVITY])
    if motion != "maneuver":
        raise ValueError("Неизвестный тип движения")
    omega = np.array([.05 * sin(.7 * t), .04 * cos(.5 * t), .15 * cos(.3 * t)])
    force = np.array([1.2 * sin(.4 * t), .9 * cos(.35 * t),
                      -GRAVITY + .3 * sin(.6 * t)])
    return omega, force


def _measurement_jacobian(kind: str, rotation: np.ndarray,
                          velocity: np.ndarray) -> np.ndarray:
    if kind == "velocity_ne":
        h = np.zeros((2, 15))
        h[:, 3:5] = np.eye(2)
        return h
    if kind == "body_velocity_xy":
        h = np.zeros((2, 15))
        h[:, 3:6] = rotation.T[0:2, :]
        h[:, 6:9] = skew(rotation.T @ velocity)[0:2, :]
        return h
    if kind == "yaw":
        r = rotation
        den = r[0, 0] ** 2 + r[1, 0] ** 2
        h = np.zeros((1, 15))
        h[0, 7] = (r[1, 0] * r[0, 2] - r[0, 0] * r[1, 2]) / den
        h[0, 8] = (r[0, 0] * r[1, 1] - r[1, 0] * r[0, 1]) / den
        return h
    if kind == "position_ne":
        h = np.zeros((2, 15))
        h[:, 0:2] = np.eye(2)
        return h
    raise ValueError("Неизвестный вид коррекции")


def observability(scenario: Scenario, *, duration_s: float = 20.,
                  imu_dt_s: float = .01, aiding_period_s: float = .1,
                  relative_tolerance: float = 1e-8) -> dict:
    """Ранг O и доли ненаблюдаемого подпространства по компонентам."""
    if not 1. <= duration_s <= 120. or not 0 < imu_dt_s <= .05:
        raise ValueError("Недопустимое окно анализа")
    q = np.array([1., 0., 0., 0.])
    v = np.array([0., 0., 0.]) if scenario.motion == "hover" else np.array([1., .3, 0.])
    phi_total = np.eye(15)
    rows = []
    steps = int(round(duration_s / imu_dt_s))
    every = max(1, int(round(aiding_period_s / imu_dt_s)))
    for k in range(steps + 1):
        t = k * imu_dt_s
        rot = rotation_body_to_ned(q)
        if k % every == 0:
            for kind in scenario.aiding:
                rows.append(_measurement_jacobian(kind, rot, v) @ phi_total)
        omega, force = _imu(t, scenario.motion)
        f = continuous_error_jacobian(rot, omega, force)
        phi_total = exp_transition(f, imu_dt_s) @ phi_total
        v = v + imu_dt_s * (rot @ force + np.array([0., 0., GRAVITY]))
        q = quat_product(q, quat_exponential(imu_dt_s * omega))
        q = q / np.linalg.norm(q)
    o = np.vstack(rows) * PRIOR_SIGMA[None, :]
    singular = np.linalg.svd(o, compute_uv=False)
    _, _, vt = np.linalg.svd(o, full_matrices=True)
    threshold = relative_tolerance * float(singular[0])
    rank = int(np.sum(singular > threshold))
    null = vt[rank:, :]
    fraction = (np.sum(null ** 2, axis=0) if null.size else np.zeros(15))
    return {
        "scenario": scenario.name,
        "motion": scenario.motion,
        "aiding": list(scenario.aiding),
        "rank": rank,
        "state_dimension": 15,
        "unobservable_dimension": 15 - rank,
        "unobservable_fraction_by_state": {
            name: round(float(value), 6) for name, value in zip(STATE_NAMES, fraction)
        },
        "unobservable_states": [
            name for name, value in zip(STATE_NAMES, fraction) if value > .5
        ],
        "window_s": duration_s,
    }


def report() -> dict:
    results = [observability(s) for s in SCENARIOS]
    return {
        "type": "LINEARIZED_ESKF15_OBSERVABILITY",
        "column_scaling": "prior sigma",
        "scenarios": results,
        "limitations": [
            "Линеаризация вдоль номинальной траектории без шумов",
            "Вертикальный канал без барометра/дальномера в РФК",
            "Конечное окно 20 с; слабая наблюдаемость не выявляется рангом",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Наблюдаемость 15-мерного РФК")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    data = report()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    for row in data["scenarios"]:
        print(f"{row['scenario']:32s} rank={row['rank']:2d} "
              f"unobservable={','.join(row['unobservable_states']) or '-'}")


if __name__ == "__main__":
    main()
