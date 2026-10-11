"""Фиксированная история инерциальных состояний с повторным прогнозом РФК.

Запаздывающее измерение визуальной скорости относится к моменту середины
экспозиции, а не к моменту получения в компьютере. После его включения
выполняется детерминированный пересчет состояния по архивным отсчетам ИИМ
и более ранним визуальным измерениям в порядке времени измерения.

Только офлайн/исследовательская реализация; для RTOS требуются
ограничение вычислительного времени, память и верификация задержек.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .error_state_inertial_filter import (
    KIND_ORDER, BodyVelocityMeasurement, EskfSnapshot, InertialErrorStateFilter,
    MapPositionMeasurement, YawMeasurement, measurement_kind,
)
from .horizontal_visual_inertial_filter import VisualVelocityMeasurement
from .imu_preintegration import ImuReading

_PROCESSED = frozenset({
    "VISUAL_CORRECTED", "VISUAL_OUTLIER_REJECTED", "YAW_CORRECTED",
    "YAW_OUTLIER_REJECTED", "POSITION_CORRECTED", "POSITION_OUTLIER_REJECTED",
})


@dataclass(frozen=True)
class ReplayResult:
    status: str
    measurement_time_us: int
    latest_imu_time_us: int
    replayed_imu_samples: int
    accepted_visual_count: int
    rejected_visual_count: int
    state: EskfSnapshot
    nis: float = float("nan")


def _interpolate(before: ImuReading, after: ImuReading, t: int) -> ImuReading:
    if not before.timestamp_us < t < after.timestamp_us:
        raise ValueError("Момент камеры вне границ двух отсчетов ИИМ")
    alpha = (t - before.timestamp_us) / (after.timestamp_us-before.timestamp_us)
    omega = np.asarray(before.gyro_body_rad_s, dtype=np.float64) * (1.0-alpha) + (
        np.asarray(after.gyro_body_rad_s, dtype=np.float64) * alpha)
    force = np.asarray(before.specific_force_body_m_s2, dtype=np.float64) * (1.0-alpha) + (
        np.asarray(after.specific_force_body_m_s2, dtype=np.float64) * alpha)
    return ImuReading(t, tuple(float(x) for x in omega), tuple(float(x) for x in force))


def _covariance_valid(value: object) -> bool:
    covariance = np.asarray(value, dtype=float)
    return bool(
        covariance.shape == (2, 2) and np.isfinite(covariance).all()
        and np.allclose(covariance, covariance.T, rtol=0, atol=1e-10)
        and np.linalg.eigvalsh(covariance)[0] > 0
    )


def _measurement_valid(measurement: object) -> bool:
    try:
        if not (isinstance(measurement.timestamp_us, int)
                and measurement.timestamp_us > 0):
            return False
        if isinstance(measurement, VisualVelocityMeasurement):
            values = (measurement.north_m_s, measurement.east_m_s)
            covariance = measurement.covariance_ne_m2_s2
        elif isinstance(measurement, BodyVelocityMeasurement):
            values = (measurement.x_m_s, measurement.y_m_s)
            covariance = measurement.covariance_xy_m2_s2
        elif isinstance(measurement, MapPositionMeasurement):
            w = float(measurement.ci_weight)
            if not np.isfinite(w) or w > 1. or (0. < w < 1. and not .01 <= w <= .99):
                return False
            values = (measurement.north_m, measurement.east_m)
            covariance = measurement.covariance_ne_m2
        elif isinstance(measurement, YawMeasurement):
            yaw, var = float(measurement.yaw_rad), float(measurement.variance_rad2)
            return bool(np.isfinite(yaw) and np.isfinite(var) and 1e-12 < var < 10.)
        else:
            return False
        velocity = np.asarray(values, dtype=float)
        return bool(velocity.shape == (2,) and np.isfinite(velocity).all()
                    and _covariance_valid(covariance))
    except (AttributeError, TypeError, ValueError, np.linalg.LinAlgError):
        return False


def _key(measurement: object) -> tuple[int, int]:
    return measurement.timestamp_us, KIND_ORDER[measurement_kind(measurement)]


class FixedLagEskfReplay:
    """История с ограниченным по времени окном, откатом и повторным прогнозом.

    Каждый ИИМ приходит в возрастающем порядке. Визуальные измерения могут
    приходить с задержкой, но не позже max_lag_us и только за интервал,
    для которого в истории есть обе опорные временные метки ИИМ.

    Фабрика гарантирует независимую инициализацию РФК. Копирование состояний
    через deepcopy допустимо для тестового офлайн-контура, не для RTOS.
    """

    def __init__(
        self,
        factory: Callable[[], InertialErrorStateFilter],
        *,
        max_lag_us: int = 300_000,
        max_imu_samples: int = 1000,
    ) -> None:
        if not 0 < max_lag_us <= 5_000_000 or max_imu_samples < 4:
            raise ValueError("Некорректный предел задержки/памяти")
        self._factory = factory
        self._filter = factory()
        self._lag_us = max_lag_us
        self._max_samples = max_imu_samples
        self._checkpoint: InertialErrorStateFilter | None = None
        self._checkpoint_imu: ImuReading | None = None
        self._raw: list[ImuReading] = []
        self._states: list[InertialErrorStateFilter] = []
        self._visual: dict[tuple[int, int], object] = {}
        self._latest_raw_time_us = 0
        self._failed = False
        self._per_measurement_status: dict[tuple[int, int], str] = {}
        self._per_measurement_nis: dict[tuple[int, int], float] = {}

    @property
    def state(self) -> EskfSnapshot:
        return self._filter.snapshot()

    @property
    def stored_imu_count(self) -> int:
        return len(self._raw)

    @property
    def stored_visual_count(self) -> int:
        return len(self._visual)

    def _trim(self) -> None:
        cutoff = self._latest_raw_time_us - self._lag_us
        while self._raw and self._raw[0].timestamp_us <= cutoff:
            self._checkpoint = self._states.pop(0)
            self._checkpoint_imu = self._raw.pop(0)
            limit = self._checkpoint_imu.timestamp_us
            self._visual = {k: m for k, m in self._visual.items() if k[0] > limit}
            self._per_measurement_status = {
                k: status for k, status in self._per_measurement_status.items()
                if k[0] > limit
            }
            self._per_measurement_nis = {
                k: value for k, value in self._per_measurement_nis.items()
                if k[0] > limit
            }

    def push_imu(self, reading: ImuReading) -> EskfSnapshot:
        if self._failed:
            raise ValueError("Инерциальный фильтр заблокирован после отказа")
        if not isinstance(reading.timestamp_us, int) or (
            reading.timestamp_us <= self._latest_raw_time_us
        ):
            raise ValueError("Повторная или немонотонная метка ИИМ")
        if self._latest_raw_time_us and (
            reading.timestamp_us - self._latest_raw_time_us >
            round(self._filter.max_imu_dt_s * 1e6)
        ):
            # Preserve failing estimator state, do not synthesize missing samples.
            self._filter.predict(reading)
            self._failed = True
            raise ValueError("Неисправимый разрыв последовательности ИИМ")
        if self._checkpoint is not None and len(self._raw) >= self._max_samples:
            raise ValueError("Окно истории ИИМ переполнено: выборка не принята")
        result = self._filter.predict(reading)
        if result.status in ("IMU_INVALID_OR_GAP", "NUMERICAL_FAILURE"):
            self._failed = True
            raise ValueError(f"Инерциальный фильтр перешел в отказ: {result.status}")
        self._latest_raw_time_us = reading.timestamp_us
        if self._checkpoint is None:
            self._checkpoint = deepcopy(self._filter)
            self._checkpoint_imu = reading
        else:
            self._raw.append(reading)
            self._states.append(deepcopy(self._filter))
        self._trim()
        return result

    def push_delayed_visual(self, measurement: object) -> ReplayResult:
        """Одно запаздывающее измерение любого поддерживаемого типа."""
        return self.push_delayed_frame([measurement])[0]

    def push_delayed_frame(self, measurements: list) -> list[ReplayResult]:
        """Все измерения одного кадра (одна экспозиция) — один пересчет.

        Каждое измерение проверяется отдельно; отклоненные не попадают в
        историю. При неудаче пересчета не сохраняется ни одно измерение.
        """
        if not measurements:
            raise ValueError("Пустой кадр")
        stamp = getattr(measurements[0], "timestamp_us", None)

        def rejected(measurement: object, status: str) -> ReplayResult:
            state = self._filter.snapshot()
            return ReplayResult(status, getattr(measurement, "timestamp_us", 0),
                                self._latest_raw_time_us, 0,
                                state.accepted_visual_count,
                                state.rejected_visual_count, state)

        results: list[ReplayResult | None] = [None] * len(measurements)
        admitted: list[tuple[int, object]] = []
        seen: set[int] = set()
        for index, measurement in enumerate(measurements):
            t = getattr(measurement, "timestamp_us", None)
            status = None
            if self._failed:
                status = "IMU_INVALID_OR_GAP"
            elif t != stamp or not _measurement_valid(measurement):
                status = "INVALID_VISUAL_MEASUREMENT"
            elif _key(measurement)[1] in seen:
                status = "INVALID_VISUAL_MEASUREMENT"
            elif self._checkpoint is None or self._checkpoint_imu is None:
                status = "NO_IMU_REFERENCE"
            elif t > self._latest_raw_time_us:
                status = "FUTURE_VISUAL_MEASUREMENT"
            elif self._latest_raw_time_us - t > self._lag_us:
                status = "VISUAL_DELAY_EXCEEDED"
            elif t <= self._checkpoint_imu.timestamp_us:
                status = "OUTSIDE_BUFFERED_HISTORY"
            elif _key(measurement) in self._visual:
                status = "REPEATED_VISUAL_MEASUREMENT"
            elif not self._raw:
                status = "INSUFFICIENT_IMU_HISTORY"
            if status is not None:
                results[index] = rejected(measurement, status)
                continue
            seen.add(_key(measurement)[1])
            admitted.append((index, measurement))
        if not admitted:
            return results  # type: ignore[return-value]

        candidate = dict(self._visual)
        for _, measurement in admitted:
            candidate[_key(measurement)] = measurement
        try:
            future_filter, future_states, statuses, niss = self._replay(candidate)
        except (ValueError, np.linalg.LinAlgError):
            for index, measurement in admitted:
                results[index] = rejected(measurement, "REPLAY_FAILED")
            return results  # type: ignore[return-value]
        # Do not persist a candidate which caused an internal correction
        # failure. NIS-based outliers are valid rejections and remain in
        # the chronological event history for reproducible diagnostics.
        if any(statuses[_key(m)] not in _PROCESSED for _, m in admitted):
            for index, measurement in admitted:
                status = statuses[_key(measurement)]
                results[index] = rejected(
                    measurement, status if status not in _PROCESSED else "REPLAY_FAILED")
            return results  # type: ignore[return-value]
        # Commit atomically only when replay has completed.
        self._visual = candidate
        self._filter = future_filter
        self._states = future_states
        self._per_measurement_status = statuses
        self._per_measurement_nis = niss
        self._trim()
        state = self._filter.snapshot()
        for index, measurement in admitted:
            key = _key(measurement)
            results[index] = ReplayResult(
                statuses[key], key[0], self._latest_raw_time_us, len(self._raw),
                state.accepted_visual_count, state.rejected_visual_count, state,
                niss[key],
            )
        return results  # type: ignore[return-value]

    def _replay(
        self, measurements: dict[tuple[int, int], object],
    ) -> tuple[
        InertialErrorStateFilter,
        list[InertialErrorStateFilter],
        dict[tuple[int, int], str],
        dict[tuple[int, int], float],
    ]:
        assert self._checkpoint is not None and self._checkpoint_imu is not None
        f = deepcopy(self._checkpoint)
        preceding = self._checkpoint_imu
        ordered = sorted(measurements.items())
        i = 0
        states: list[InertialErrorStateFilter] = []
        statuses: dict[tuple[int, int], str] = {}
        niss: dict[tuple[int, int], float] = {}

        def apply_all_at(stamp: int) -> None:
            nonlocal i
            while i < len(ordered) and ordered[i][0][0] == stamp:
                key, measurement = ordered[i]
                result = f.apply(measurement)
                statuses[key] = result.status
                niss[key] = result.last_nis
                i += 1

        for raw in self._raw:
            while i < len(ordered) and ordered[i][0][0] < raw.timestamp_us:
                stamp = ordered[i][0][0]
                if stamp <= preceding.timestamp_us:
                    raise ValueError("Визуальное измерение до начала истории")
                f.predict(_interpolate(preceding, raw, stamp))
                if f.snapshot().status in ("IMU_INVALID_OR_GAP", "NUMERICAL_FAILURE"):
                    raise ValueError("Прогноз завершился отказом")
                apply_all_at(stamp)
            f.predict(raw)
            if f.snapshot().status in ("IMU_INVALID_OR_GAP", "NUMERICAL_FAILURE"):
                raise ValueError("Повторный прогноз завершился отказом")
            apply_all_at(raw.timestamp_us)
            states.append(deepcopy(f))
            preceding = raw
        if i != len(ordered):
            raise ValueError("Визуальное измерение вне истории ИИМ")
        return f, states, statuses, niss
