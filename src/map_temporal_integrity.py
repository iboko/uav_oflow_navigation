"""Temporal integrity screening for orthophoto position candidates.

A chain of geometrically plausible images is not a certified absolute fix:
a repeated feature can remain consistently wrong over many frames.
This gate only detects impossible motion, large jumps, timestamps and
insufficient temporal support. It never changes the ESKF/PX4 state.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite

from .orthophoto_localization import MapLocalizationFix


@dataclass(frozen=True)
class TemporalMapAssessment:
    status: str
    consistent: bool
    confirmed_intervals: int
    apparent_speed_m_s: float = float("nan")
    usable_as_eskf_observation: bool = False


class TemporalMapIntegrityMonitor:
    """Fail-closed temporal screening of sequential map candidates.

    Without VO/reference velocity this tests physically possible motion,
    not correctness. Lock is permanently invalid after contradictory
    movement until reset() to prevent silent re-acquisition.
    """

    def __init__(
        self,
        *,
        max_horizontal_speed_m_s: float = 8.,
        max_interframe_gap_s: float = 0.5,
        minimum_consistent_intervals: int = 3,
    ) -> None:
        if not (isfinite(max_horizontal_speed_m_s)
                and 0 < max_horizontal_speed_m_s <= 30):
            raise ValueError("Неверный предел горизонтальной скорости")
        if not (isfinite(max_interframe_gap_s)
                and 0 < max_interframe_gap_s <= 5):
            raise ValueError("Неверный допустимый промежуток между кадрами")
        if not 1 <= minimum_consistent_intervals <= 100:
            raise ValueError("Недопустимое число подтверждающих интервалов")
        self._max_speed = max_horizontal_speed_m_s
        self._max_gap = max_interframe_gap_s
        self._min_intervals = minimum_consistent_intervals
        self.reset()

    def reset(self) -> None:
        """Explicit reset only; never silently recover after integrity failure."""
        self._previous: tuple[int, float, float] | None = None
        self._n = 0
        self._fault = False

    def inspect(self, timestamp_us: int, fix: MapLocalizationFix) -> TemporalMapAssessment:
        def report(status: str, consistent: bool, speed: float = float("nan")):
            return TemporalMapAssessment(status, consistent, self._n, speed, False)

        if self._fault:
            return report("MAP_INTEGRITY_LATCHED", False)
        if not isinstance(timestamp_us, int) or timestamp_us <= 0:
            self._fault = True
            return report("MAP_TIMESTAMP_INVALID", False)
        if not fix.accepted or not all(isfinite(x) for x in
                                       (fix.north_m, fix.east_m)):
            # Incomplete observation loses temporal continuity. Starting
            # a fresh lock requires explicit reset, not hidden bridging.
            self._fault = True
            return report("MAP_OBSERVATION_INTERRUPTED", False)
        if self._previous is None:
            self._previous = (timestamp_us, fix.north_m, fix.east_m)
            return report("MAP_HISTORY_INITIALIZED", False)
        old_t, old_n, old_e = self._previous
        dt = (timestamp_us - old_t) * 1e-6
        if dt <= 0:
            self._fault = True
            return report("MAP_TIMESTAMP_OUT_OF_ORDER", False)
        if dt > self._max_gap:
            self._fault = True
            return report("MAP_TIME_GAP", False)
        speed = hypot(fix.north_m-old_n, fix.east_m-old_e) / dt
        if not isfinite(speed) or speed > self._max_speed:
            self._fault = True
            return report("MAP_KINEMATIC_INCONSISTENCY", False, speed)
        self._n += 1
        self._previous = (timestamp_us, fix.north_m, fix.east_m)
        if self._n < self._min_intervals:
            return report("MAP_TEMPORAL_CONFIRMATION_PENDING", False, speed)
        return report("MAP_TEMPORALLY_CONSISTENT_UNVALIDATED", True, speed)
