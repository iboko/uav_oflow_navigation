from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Optional

import numpy as np

try:
    from config import NavigationConfig
except ImportError:
    from .config import NavigationConfig


@dataclass
class SafetyStatus:
    mode: str
    warnings: list[str] = field(default_factory=list)
    allow_optical_flow: bool = True


class OpticalFlowSafetyMonitor:
    """Контроль применимости режима ОП.

    Класс не управляет БВС напрямую. Он формирует диагностический статус,
    который затем используется контуром САУ/автопилотом для запрета режима,
    перехода в посадку, зависание или ручной режим.
    """

    def __init__(self, cfg: NavigationConfig):
        self.cfg = cfg
        self.last_flow_time_s: Optional[float] = None
        self.last_range_time_s: Optional[float] = None
        self.last_good_flow_time_s: Optional[float] = None

    def update(
        self,
        t_s: float,
        range_m: float,
        height_m: float,
        flow_quality: float,
        imu_valid: bool,
        flow_received: bool,
        range_received: bool,
    ) -> SafetyStatus:
        warnings: list[str] = []

        if range_received and np.isfinite(range_m):
            self.last_range_time_s = t_s
        if flow_received:
            self.last_flow_time_s = t_s

        if flow_received and np.isfinite(flow_quality) and flow_quality >= self.cfg.quality_min:
            self.last_good_flow_time_s = t_s

        if not imu_valid:
            warnings.append("IMU_INVALID")

        if self.last_range_time_s is None or (t_s - self.last_range_time_s) > self.cfg.range_stale_timeout_s:
            warnings.append("RANGE_STALE")

        if self.last_flow_time_s is None or (t_s - self.last_flow_time_s) > self.cfg.flow_stale_timeout_s:
            warnings.append("FLOW_STALE")

        if np.isfinite(height_m):
            if height_m < self.cfg.min_height_m:
                warnings.append("HEIGHT_TOO_LOW")
            elif height_m > self.cfg.max_height_m:
                warnings.append("HEIGHT_TOO_HIGH")
        else:
            warnings.append("HEIGHT_INVALID")

        if flow_received and np.isfinite(flow_quality) and flow_quality < self.cfg.quality_min:
            warnings.append("LOW_FLOW_QUALITY")

        if self.last_good_flow_time_s is None:
            warnings.append("NO_GOOD_FLOW_YET")
        elif (t_s - self.last_good_flow_time_s) > 0.50:
            warnings.append("FLOW_DEGRADED_LONG")

        critical = {"IMU_INVALID", "RANGE_STALE", "HEIGHT_INVALID", "HEIGHT_TOO_LOW", "HEIGHT_TOO_HIGH"}
        degraded = {"FLOW_STALE", "LOW_FLOW_QUALITY", "FLOW_DEGRADED_LONG", "NO_GOOD_FLOW_YET"}

        if any(w in critical for w in warnings):
            return SafetyStatus("FAILSAFE_LAND_OR_MANUAL", warnings, allow_optical_flow=False)

        if any(w in degraded for w in warnings):
            return SafetyStatus("OP_DEGRADED_HOLD_OR_BRAKE", warnings, allow_optical_flow=False)

        return SafetyStatus("OP_VALID", warnings, allow_optical_flow=True)
