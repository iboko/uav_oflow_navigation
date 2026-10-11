"""MSCKF на компаньоне: ИИМ автопилота + кадры → скорость для EKF2.

Вход: отсчеты ИИМ автопилота (HIGHRES_IMU, уже на часах компаньона),
кадры с признаками, ориентация AHRS и дальность дальномера на моменты
кадров (из ``CompanionVo``). Без ГНСС:

1. накопление окна; не раньше 1,5 с и не ниже ``min_height_m`` —
   выставка ``vi_initialization.initialize`` с дальностью до земли
   (дальномер × cos наклона, допущение ровной земли);
2. при успехе — MSCKF с конца окна;
3. на каждый кадр — скорость в осях корпуса FRD (R^T v) и ее
   ковариация. Курс MSCKF ненаблюдаем, скорость в осях корпуса от него
   не зависит;
4. расхождение (СКО скорости > ``max_sigma_v``, скорость > ``max_speed``
   или численный отказ) — сброс и новая выставка, без вывода;
5. разрыв ИИМ до 0,1 с (потерянные пакеты MAVLink) заполняется линейной
   интерполяцией и считается (``IMU_GAP_BRIDGED``), длиннее — сброс
   (``RESET_IMU_GAP``).
"""
from __future__ import annotations

from dataclasses import dataclass
from math import cos

import numpy as np

from .error_state_inertial_filter import ImuNoiseDensity
from .imu_preintegration import ImuBias, ImuReading, rotation_body_to_ned
from .msckf_vio import Msckf, PinholeCamera
from .vi_initialization import initialize


@dataclass(frozen=True)
class MsckfVelocity:
    time_us: int
    velocity_frd: tuple[float, float, float]
    variance_frd: tuple[float, float, float]
    velocity_ned_msckf: tuple[float, float, float]


class CompanionMsckf:
    def __init__(self, camera: PinholeCamera, noise: ImuNoiseDensity, *, max_clones: int = 10,
                 min_window_s: float = 1.5, max_window_s: float = 5.0, min_height_m: float = 3.0,
                 max_sigma_v: float = 1.0, max_speed_m_s: float = 25.0, max_imu_step_s: float = 0.02,
                 max_imu_gap_s: float = 0.1) -> None:
        self.camera = camera
        self.noise = noise
        self.max_clones = max_clones
        self.min_window_s = min_window_s
        self.max_window_s = max_window_s
        self.min_height = min_height_m
        self.max_sigma_v = max_sigma_v
        self.max_speed = max_speed_m_s
        self.max_imu_step_us = int(max_imu_step_s * 1e6)
        self.max_imu_gap_us = int(max_imu_gap_s * 1e6)
        self.imu: list[ImuReading] = []
        self.filter: Msckf | None = None
        self._window: list[tuple[int, dict]] = []
        self._heights: dict[int, float] = {}
        self._q0 = None
        self._last_try = 0
        self.counts: dict[str, int] = {}
        self.last_init: dict | None = None
        self.last_error: str | None = None

    def count(self, key: str) -> None:
        self.counts[key] = self.counts.get(key, 0) + 1

    def add_imu(self, reading: ImuReading) -> None:
        if self.imu and reading.timestamp_us <= self.imu[-1].timestamp_us:
            return
        if self.imu:
            last = self.imu[-1]
            gap = reading.timestamp_us - last.timestamp_us
            if gap > self.max_imu_gap_us:
                # Lost IMU data: neither the window nor the filter can be
                # propagated through it.
                self.imu.clear()
                if self.filter is not None or self._window:
                    self.reset("RESET_IMU_GAP")
            elif gap > self.max_imu_step_us:
                # A dropped MAVLink packet or two (56 ms seen in SITL at
                # 125 Hz): linear interpolation of the rates, i.e. the same
                # trapezoidal integration over a longer step. Counted.
                n = int(np.ceil(gap / self.max_imu_step_us))
                for k in range(1, n):
                    self.imu.append(self._interpolate(last, reading, last.timestamp_us + k * gap // n))
                self.count("IMU_GAP_BRIDGED")
        self.imu.append(reading)
        # Keep ~8 s: enough for the longest initialisation window.
        cut = reading.timestamp_us - 8_000_000
        while self.imu and self.imu[0].timestamp_us < cut:
            self.imu.pop(0)

    @staticmethod
    def _interpolate(a: ImuReading, b: ImuReading, t_us: int) -> ImuReading:
        w = (t_us - a.timestamp_us) / (b.timestamp_us - a.timestamp_us)
        mix = lambda x, y: tuple((1 - w) * np.asarray(x) + w * np.asarray(y))  # noqa: E731
        return ImuReading(t_us, mix(a.gyro_body_rad_s, b.gyro_body_rad_s),
                          mix(a.specific_force_body_m_s2, b.specific_force_body_m_s2))

    def _reading_at(self, t_us: int) -> ImuReading | None:
        ts = [r.timestamp_us for r in self.imu]
        i = int(np.searchsorted(ts, t_us))
        if i < len(ts) and ts[i] == t_us:
            return self.imu[i]
        if i == 0 or i == len(ts):
            return None
        return self._interpolate(self.imu[i - 1], self.imu[i], t_us)

    def ready_for(self, t_us: int) -> bool:
        return bool(self.imu) and self.imu[-1].timestamp_us >= t_us

    def reset(self, reason: str) -> None:
        self.filter = None
        self._window.clear()
        self._heights.clear()
        self._q0 = None
        self.count(reason)

    def process(self, t_us: int, features: dict[int, tuple[float, float]], *, attitude_q,
                roll: float, pitch: float, range_m: float) -> MsckfVelocity | None:
        """t_us on the companion clock; attitude_q = AHRS (autopilot) at the frame."""
        if not self.ready_for(t_us):
            self.count("NO_IMU_AFTER_FRAME")
            return None
        height = range_m * cos(roll) * cos(pitch)
        if self.filter is None:
            if not height >= self.min_height:
                self._window.clear()
                self._heights.clear()
                self.count("BELOW_MIN_HEIGHT")
                return None
            if not self._window:
                self._q0 = attitude_q
            self._window.append((t_us, features))
            self._heights[t_us] = height
            span = (t_us - self._window[0][0]) * 1e-6
            if span < self.min_window_s or t_us - self._last_try < 500_000:
                return None
            self._last_try = t_us
            try:
                res = initialize(self.imu, self._window, self.camera, self._q0, heights_agl_m=self._heights)
            except (ValueError, np.linalg.LinAlgError) as exc:
                self.last_error = f"init {type(exc).__name__}: {exc}"
                self.reset("INIT_ERROR")
                return None
            self.last_init = {"status": res.status, "window_s": round(span, 2),
                              "sigma_velocity_m_s": res.sigma_velocity_m_s, "features": res.features,
                              "median_depth_m": res.median_depth_m, "chi2_per_dof": res.chi2_per_dof,
                              "gravity_tilt_deg": list(res.gravity_tilt_deg),
                              "velocity_end": list(res.velocity_end_ned_m_s)}
            self.count(f"INIT_{res.status}")
            if not res.ok:
                if span > self.max_window_s:
                    self.reset("INIT_WINDOW_RESTART")
                return None
            sv = np.sqrt(np.clip(np.diag(np.asarray(res.velocity_end_cov)), 1e-4, None))
            sig = (0.01, 0.01, 0.01, *sv, 0.02, 0.02, 0.1, 0.01, 0.01, 0.01,
                   *(max(res.sigma_accel_bias_m_s2, 0.02),) * 3)
            self.filter = Msckf(camera=self.camera, noise=self.noise, quaternion_body_to_ned=res.quaternion_end,
                                initial_error_sigma=sig, initial_velocity_ned_m_s=res.velocity_end_ned_m_s,
                                initial_bias=ImuBias(accel_body_m_s2=res.accel_bias_body_m_s2),
                                max_clones=self.max_clones)
            self.filter.predict(self._reading_at(t_us))
            self._window.clear()
            self._heights.clear()
            return None
        f = self.filter
        try:
            for r in self.imu:
                if f.last_imu.timestamp_us < r.timestamp_us < t_us:
                    f.predict(r)
            if f.last_imu.timestamp_us != t_us:
                f.predict(self._reading_at(t_us))
            f.process_frame(t_us, features)
        except (ValueError, np.linalg.LinAlgError) as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.reset("RESET_NUMERICAL")
            return None
        r_nb = rotation_body_to_ned(f.q)
        cov_v = f.cov[3:6, 3:6]
        if (not np.isfinite(f.v).all() or np.linalg.norm(f.v) > self.max_speed
                or np.sqrt(np.max(np.linalg.eigvalsh(cov_v))) > self.max_sigma_v):
            self.reset("RESET_DIVERGED")
            return None
        v_b = r_nb.T @ f.v
        cov_b = r_nb.T @ cov_v @ r_nb
        self.count("PUBLISHED")
        return MsckfVelocity(t_us, tuple(float(x) for x in v_b), tuple(float(cov_b[i, i]) for i in range(3)),
                             tuple(float(x) for x in f.v))
