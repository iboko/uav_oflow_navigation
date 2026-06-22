from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class NavigationConfig:
    """Параметры алгоритма ОП-навигации.

    Значения выбраны как безопасный старт для маловысотного режима
    на мультироторном БВС. Для реального борта их нужно уточнять
    по датчику, прошивке и результатам логов.
    """

    imu_rate_hz: float = 100.0
    flow_rate_hz: float = 25.0

    min_height_m: float = 0.20
    max_height_m: float = 4.00
    quality_min: int = 100

    flow_stale_timeout_s: float = 0.20
    range_stale_timeout_s: float = 0.20

    max_abs_flow_rate_rad_s: float = 7.4
    max_horizontal_speed_mps: float = 1.5

    base_flow_vel_sigma_mps: float = 0.035
    accel_noise_sigma_mps2: float = 0.08
    gyro_noise_sigma_radps: float = 0.004
    range_noise_sigma_m: float = 0.015

    @classmethod
    def from_yaml(cls, path: str | Path) -> "NavigationConfig":
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        allowed = {field.name for field in cls.__dataclass_fields__.values()}
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValueError(f"Неизвестные параметры конфигурации: {unknown}")
        return cls(**data)
