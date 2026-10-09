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

from .error_state_inertial_filter import InertialErrorStateFilter, EskfSnapshot
from .horizontal_visual_inertial_filter import VisualVelocityMeasurement
from .imu_preintegration import ImuReading


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


def _measurement_valid(measurement: VisualVelocityMeasurement) -> bool:
    try:
        velocity = np.asarray((measurement.north_m_s, measurement.east_m_s), dtype=float)
        covariance = np.asarray(measurement.covariance_ne_m2_s2, dtype=float)
        return bool(
            isinstance(measurement.timestamp_us, int) and measurement.timestamp_us > 0
            and velocity.shape == (2,) and np.isfinite(velocity).all()
            and covariance.shape == (2, 2) and np.isfinite(covariance).all()
            and np.allclose(covariance, covariance.T, rtol=0, atol=1e-10)
            and np.linalg.eigvalsh(covariance)[0] > 0
        )
    except (TypeError, ValueError, np.linalg.LinAlgError):
        return False


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
        self._visual: dict[int, VisualVelocityMeasurement] = {}
        self._latest_raw_time_us = 0
        self._failed = False
        self._per_measurement_status: dict[int, str] = {}
        self._per_measurement_nis: dict[int, float] = {}

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
            self._visual = {t: m for t, m in self._visual.items()
                            if t > self._checkpoint_imu.timestamp_us}
            self._per_measurement_status = {
                t: status for t, status in self._per_measurement_status.items()
                if t > self._checkpoint_imu.timestamp_us
            }
            self._per_measurement_nis = {
                t: value for t, value in self._per_measurement_nis.items()
                if t > self._checkpoint_imu.timestamp_us
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

    def push_delayed_visual(
        self, measurement: VisualVelocityMeasurement,
    ) -> ReplayResult:
        t = measurement.timestamp_us
        def rejected(status: str) -> ReplayResult:
            state = self._filter.snapshot()
            return ReplayResult(status, t, self._latest_raw_time_us, 0,
                                state.accepted_visual_count,
                                state.rejected_visual_count, state)

        if self._failed:
            return rejected("IMU_INVALID_OR_GAP")
        if not _measurement_valid(measurement):
            return rejected("INVALID_VISUAL_MEASUREMENT")
        if self._checkpoint is None or self._checkpoint_imu is None:
            return rejected("NO_IMU_REFERENCE")
        if t > self._latest_raw_time_us:
            return rejected("FUTURE_VISUAL_MEASUREMENT")
        if self._latest_raw_time_us - t > self._lag_us:
            return rejected("VISUAL_DELAY_EXCEEDED")
        if t <= self._checkpoint_imu.timestamp_us:
            return rejected("OUTSIDE_BUFFERED_HISTORY")
        if t in self._visual:
            return rejected("REPEATED_VISUAL_MEASUREMENT")
        if not self._raw:
            return rejected("INSUFFICIENT_IMU_HISTORY")

        candidate = dict(self._visual)
        candidate[t] = measurement
        try:
            future_filter, future_states, statuses, niss = self._replay(candidate)
        except (ValueError, np.linalg.LinAlgError):
            return rejected("REPLAY_FAILED")
        # Do not persist a candidate which caused an internal correction
        # failure. NIS-based outliers are valid rejections and remain in
        # the chronological event history for reproducible diagnostics.
        if statuses[t] not in ("VISUAL_CORRECTED", "VISUAL_OUTLIER_REJECTED"):
            return rejected(statuses[t])
        # Commit atomically only when replay has completed.
        self._visual = candidate
        self._filter = future_filter
        self._states = future_states
        self._per_measurement_status = statuses
        self._per_measurement_nis = niss
        self._trim()
        state = self._filter.snapshot()
        return ReplayResult(
            statuses[t], t, self._latest_raw_time_us, len(self._raw),
            state.accepted_visual_count, state.rejected_visual_count, state,
            niss[t],
        )

    def _replay(
        self, measurements: dict[int, VisualVelocityMeasurement],
    ) -> tuple[
        InertialErrorStateFilter,
        list[InertialErrorStateFilter],
        dict[int, str],
        dict[int, float],
    ]:
        assert self._checkpoint is not None and self._checkpoint_imu is not None
        f = deepcopy(self._checkpoint)
        preceding = self._checkpoint_imu
        ordered = sorted(measurements.values(), key=lambda x: x.timestamp_us)
        i = 0
        states: list[InertialErrorStateFilter] = []
        statuses: dict[int, str] = {}
        niss: dict[int, float] = {}
        for raw in self._raw:
            while i < len(ordered) and ordered[i].timestamp_us < raw.timestamp_us:
                visual = ordered[i]
                stamp = visual.timestamp_us
                if stamp <= preceding.timestamp_us:
                    raise ValueError("Визуальное измерение до начала истории")
                f.predict(_interpolate(preceding, raw, stamp))
                if f.snapshot().status in ("IMU_INVALID_OR_GAP", "NUMERICAL_FAILURE"):
                    raise ValueError("Прогноз завершился отказом")
                result = f.update_visual_velocity(visual)
                statuses[stamp] = result.status
                niss[stamp] = result.last_nis
                i += 1
            f.predict(raw)
            if f.snapshot().status in ("IMU_INVALID_OR_GAP", "NUMERICAL_FAILURE"):
                raise ValueError("Повторный прогноз завершился отказом")
            while i < len(ordered) and ordered[i].timestamp_us == raw.timestamp_us:
                visual = ordered[i]
                result = f.update_visual_velocity(visual)
                statuses[visual.timestamp_us] = result.status
                niss[visual.timestamp_us] = result.last_nis
                i += 1
            states.append(deepcopy(f))
            preceding = raw
        if i != len(ordered):
            raise ValueError("Визуальное измерение вне истории ИИМ")
        return f, states, statuses, niss
