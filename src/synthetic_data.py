from __future__ import annotations

import argparse
from pathlib import Path
import math

import numpy as np
import pandas as pd

try:
    from config import NavigationConfig
    from oflow_math import velocity_to_optical_flow_rad
except ImportError:
    from .config import NavigationConfig
    from .oflow_math import velocity_to_optical_flow_rad


def _central_derivative(y: np.ndarray, dt: float) -> np.ndarray:
    dy = np.zeros_like(y)
    dy[1:-1] = (y[2:] - y[:-2]) / (2.0 * dt)
    dy[0] = (y[1] - y[0]) / dt
    dy[-1] = (y[-1] - y[-2]) / dt
    return dy


def generate_synthetic_log(
    duration_s: float = 90.0,
    seed: int = 42,
    cfg: NavigationConfig | None = None,
) -> pd.DataFrame:
    """Генерирует реалистичный лог маловысотного полета по ОП.

    Сценарий:
    - высота 0.8...1.2 м;
    - малые горизонтальные скорости до 0.25 м/с;
    - зависание с медленными перемещениями;
    - участки деградации: слабая текстура, вибрация, потеря дальномера, плохая видимость.
    """
    cfg = cfg or NavigationConfig()
    rng = np.random.default_rng(seed)

    imu_dt = 1.0 / cfg.imu_rate_hz
    flow_dt = 1.0 / cfg.flow_rate_hz

    t = np.arange(0.0, duration_s + imu_dt / 2.0, imu_dt)
    n = len(t)

    # Истинная скорость: комбинация медленных синусоид и коротких команд оператора.
    true_vn = (
        0.11 * np.sin(2.0 * np.pi * t / 18.0)
        + 0.05 * np.sin(2.0 * np.pi * t / 7.5)
    )
    true_ve = (
        0.10 * np.cos(2.0 * np.pi * t / 23.0)
        + 0.04 * np.sin(2.0 * np.pi * t / 11.0)
    )

    # Командный участок "сдвиг вправо-вперед".
    true_vn += np.where((t > 25.0) & (t < 35.0), 0.12, 0.0)
    true_ve += np.where((t > 36.0) & (t < 46.0), -0.10, 0.0)

    # Ограничение скорости как на безопасном первом этапе.
    speed = np.hypot(true_vn, true_ve)
    scale = np.minimum(1.0, 0.32 / np.maximum(speed, 1e-9))
    true_vn *= scale
    true_ve *= scale

    true_n = np.cumsum(true_vn) * imu_dt
    true_e = np.cumsum(true_ve) * imu_dt

    true_alt = 1.05 + 0.10 * np.sin(2.0 * np.pi * t / 31.0) + 0.03 * np.sin(2.0 * np.pi * t / 8.0)
    true_alt = np.clip(true_alt, 0.65, 1.45)

    roll = 0.035 * np.sin(2.0 * np.pi * t / 9.0) + 0.015 * np.sin(2.0 * np.pi * t / 3.7)
    pitch = 0.032 * np.cos(2.0 * np.pi * t / 10.5) + 0.010 * np.sin(2.0 * np.pi * t / 4.3)
    yaw = 0.18 * np.sin(2.0 * np.pi * t / 55.0)

    gyro_x_true = _central_derivative(roll, imu_dt)
    gyro_y_true = _central_derivative(pitch, imu_dt)
    gyro_z_true = _central_derivative(yaw, imu_dt)

    acc_n_true = _central_derivative(true_vn, imu_dt)
    acc_e_true = _central_derivative(true_ve, imu_dt)

    # Перевод горизонтального ускорения в связанную СК.
    c = np.cos(yaw)
    s = np.sin(yaw)
    acc_x_body_true = c * acc_n_true + s * acc_e_true
    acc_y_body_true = -s * acc_n_true + c * acc_e_true

    # Медленно меняющиеся смещения IMU.
    acc_bias_x = 0.018 * np.sin(2.0 * np.pi * t / 70.0)
    acc_bias_y = -0.014 * np.cos(2.0 * np.pi * t / 63.0)

    vibration = ((t > 60.0) & (t < 65.0)).astype(float)
    acc_noise_sigma = cfg.accel_noise_sigma_mps2 * (1.0 + 3.0 * vibration)
    gyro_noise_sigma = cfg.gyro_noise_sigma_radps * (1.0 + 3.5 * vibration)

    acc_x = acc_x_body_true + acc_bias_x + rng.normal(0.0, acc_noise_sigma)
    acc_y = acc_y_body_true + acc_bias_y + rng.normal(0.0, acc_noise_sigma)
    gyro_x = gyro_x_true + rng.normal(0.0, gyro_noise_sigma)
    gyro_y = gyro_y_true + rng.normal(0.0, gyro_noise_sigma)
    gyro_z = gyro_z_true + rng.normal(0.0, gyro_noise_sigma)

    range_m = true_alt / (np.cos(roll) * np.cos(pitch))
    range_m += rng.normal(0.0, cfg.range_noise_sigma_m, size=n)

    # Потеря дальномера.
    range_dropout = (t > 70.0) & (t < 73.0)
    range_m[range_dropout] = np.nan

    texture_class = np.full(n, "textured_floor", dtype=object)
    texture_class[(t > 45.0) & (t < 52.0)] = "low_texture"
    texture_class[(t > 80.0) & (t < 84.0)] = "smoke_or_darkness"
    texture_class[(t > 60.0) & (t < 65.0)] = "vibration"

    light_lux = np.full(n, 450.0)
    light_lux[(t > 80.0) & (t < 84.0)] = 60.0
    light_lux += rng.normal(0.0, 15.0, size=n)
    light_lux = np.clip(light_lux, 5.0, None)

    quality = np.full(n, np.nan)
    of_dt_s = np.full(n, np.nan)
    of_ix = np.full(n, np.nan)
    of_iy = np.full(n, np.nan)
    of_ixg = np.full(n, np.nan)
    of_iyg = np.full(n, np.nan)
    of_izg = np.full(n, np.nan)

    flow_step = max(1, int(round(cfg.imu_rate_hz / cfg.flow_rate_hz)))
    for i in range(0, n, flow_step):
        ti = t[i]

        q = 215.0 + rng.normal(0.0, 12.0)
        flow_noise_rad = 0.0010

        if 45.0 < ti < 52.0:
            q = 65.0 + rng.normal(0.0, 20.0)
            flow_noise_rad = 0.0060
        elif 60.0 < ti < 65.0:
            q = 155.0 + rng.normal(0.0, 25.0)
            flow_noise_rad = 0.0040
        elif 80.0 < ti < 84.0:
            q = 45.0 + rng.normal(0.0, 18.0)
            flow_noise_rad = 0.0080

        q = float(np.clip(q, 0.0, 255.0))
        quality[i] = q
        of_dt_s[i] = flow_dt

        vals = velocity_to_optical_flow_rad(
            vx_nav_mps=float(true_vn[i]),
            vy_nav_mps=float(true_ve[i]),
            yaw_rad=float(yaw[i]),
            height_m=float(true_alt[i]),
            dt_s=flow_dt,
            gyro_x_radps=float(gyro_x[i]),
            gyro_y_radps=float(gyro_y[i]),
            gyro_z_radps=float(gyro_z[i]),
            flow_noise_rad=flow_noise_rad,
            rng=rng,
        )
        of_ix[i], of_iy[i], of_ixg[i], of_iyg[i], of_izg[i] = vals

        # Иногда имитируем пропуск кадра ОП.
        if 0.0 < rng.random() < 0.005:
            quality[i] = np.nan
            of_dt_s[i] = np.nan
            of_ix[i] = of_iy[i] = of_ixg[i] = of_iyg[i] = of_izg[i] = np.nan

    # ГНСС есть только до входа в зону деградации/помещение.
    gps_available = t < 18.0
    gps_vn = np.where(gps_available, true_vn + rng.normal(0.0, 0.06, size=n), np.nan)
    gps_ve = np.where(gps_available, true_ve + rng.normal(0.0, 0.06, size=n), np.nan)

    df = pd.DataFrame(
        {
            "t_s": t,
            "true_n_m": true_n,
            "true_e_m": true_e,
            "true_vn_mps": true_vn,
            "true_ve_mps": true_ve,
            "true_alt_m": true_alt,
            "roll_rad": roll,
            "pitch_rad": pitch,
            "yaw_rad": yaw,
            "gyro_x_radps": gyro_x,
            "gyro_y_radps": gyro_y,
            "gyro_z_radps": gyro_z,
            "acc_x_mps2": acc_x,
            "acc_y_mps2": acc_y,
            "range_m": range_m,
            "of_dt_s": of_dt_s,
            "of_integrated_x_rad": of_ix,
            "of_integrated_y_rad": of_iy,
            "of_integrated_xgyro_rad": of_ixg,
            "of_integrated_ygyro_rad": of_iyg,
            "of_integrated_zgyro_rad": of_izg,
            "of_quality": quality,
            "texture_class": texture_class,
            "light_lux": light_lux,
            "gps_available": gps_available.astype(int),
            "gps_vn_mps": gps_vn,
            "gps_ve_mps": gps_ve,
        }
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Генератор синтетического лога БВС по оптическому потоку")
    parser.add_argument("--out", default="data/synthetic_oflow_log.csv", help="Путь CSV")
    parser.add_argument("--duration", type=float, default=90.0, help="Длительность, с")
    parser.add_argument("--seed", type=int, default=42, help="Seed генератора")
    parser.add_argument("--config", default=None, help="YAML-конфигурация")
    args = parser.parse_args()

    cfg = NavigationConfig.from_yaml(args.config) if args.config else NavigationConfig()
    df = generate_synthetic_log(args.duration, args.seed, cfg)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Сохранено: {out} ({len(df)} строк)")


if __name__ == "__main__":
    main()
