"""Бортовой конвейер VO на компаньон-компьютере: кадры -> скорость для EKF2.

Вход: кадры нижней камеры (метка — середина экспозиции, часы
компаньона), ориентация и дальность дальномера от автопилота (часы
автопилота). Шкалы связывает смещение, оцененное по обмену MAVLink
TIMESYNC. Выход: трехмерная скорость в осях корпуса FRD с дисперсиями,
метка — середина интервала пары кадров (часы компаньона; автопилот
переводит ее по своей синхронизации).

Горизонтальная скорость — ContinuousPlanarOdometry (плоская местность,
калиброванная камера). Вертикальная — скорость изменения высоты по
дальномеру на том же интервале: это измерение, а не подставленное
значение (допущение ровной земли между двумя отсчетами). Без высоты
по обе стороны интервала измерение не выдается.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass
from math import cos, isfinite

import numpy as np

from .attitude_time_alignment import quaternion_to_rpy, slerp
from .camera_rotation_compensation import body_to_ned
from .continuous_visual_odometry import ContinuousPlanarOdometry, VisualFrame, VisualOdomResult


class TimeOffset:
    """t_автопилота - t_компаньона по обмену TIMESYNC (нс в сообщениях).

    Запрос компаньона: tc1=0, ts1=t_c(отправки). Ответ автопилота:
    tc1=t_a, ts1 без изменений. Смещение = t_a - (ts1 + t_c(приема))/2;
    берется отсчет с минимальной задержкой круга из последних window.
    """

    def __init__(self, window: int = 20, max_rtt_us: int = 50_000) -> None:
        self._samples: list[tuple[int, int]] = []
        self._window = window
        self._max_rtt = max_rtt_us

    def add(self, sent_us: int, remote_us: int, received_us: int) -> None:
        rtt = received_us - sent_us
        if rtt < 0 or rtt > self._max_rtt:
            return
        self._samples.append((rtt, remote_us - (sent_us + received_us) // 2))
        del self._samples[:-self._window]

    @property
    def valid(self) -> bool:
        return len(self._samples) >= 3

    @property
    def offset_us(self) -> int:
        if not self.valid:
            raise ValueError("Смещение шкал времени еще не оценено")
        return min(self._samples)[1]


class History:
    """Отсчеты автопилота (часы автопилота), интерполяция без экстраполяции."""

    def __init__(self, max_gap_us: int, keep_us: int = 5_000_000) -> None:
        self._t: list[int] = []
        self._v: list = []
        self._max_gap = max_gap_us
        self._keep = keep_us

    def add(self, t_us: int, value) -> None:
        if self._t and t_us <= self._t[-1]:
            return
        self._t.append(t_us)
        self._v.append(value)
        while self._t and self._t[0] < t_us - self._keep:
            self._t.pop(0)
            self._v.pop(0)

    @property
    def latest_us(self) -> int:
        return self._t[-1] if self._t else 0

    def bracket(self, t_us: int):
        i = bisect.bisect_left(self._t, t_us)
        if i < len(self._t) and self._t[i] == t_us:
            return self._v[i], self._v[i], 0.0
        if i == 0 or i == len(self._t) or self._t[i] - self._t[i - 1] > self._max_gap:
            return None
        return self._v[i - 1], self._v[i], (t_us - self._t[i - 1]) / (self._t[i] - self._t[i - 1])


@dataclass(frozen=True)
class BodyVelocityMeasurement:
    time_us: int                 # середина интервала, часы компаньона
    velocity_frd: tuple[float, float, float]
    variance_frd: tuple[float, float, float]
    vo: VisualOdomResult
    vd_m_s: float
    height_mid_m: float = float("nan")   # над ровной землей, середина интервала


class CompanionVo:
    def __init__(self, odometry: ContinuousPlanarOdometry, *, height_sigma_m: float = 0.02,
                 max_tilt_rad: float = 0.45, min_range_rate_dt_s: float = 0.02,
                 min_height_m: float = 0.5) -> None:
        self.vo = odometry
        self.attitude = History(max_gap_us=50_000)   # кватернион FRD->NED
        self.distance = History(max_gap_us=150_000)  # дальность по оси дальномера, м
        self.clock = TimeOffset()
        self._sigma_h = height_sigma_m
        self._max_tilt = max_tilt_rad
        self._min_dt = min_range_rate_dt_s
        # Below this the footprint is a few cm and the image blurs on
        # takeoff/landing: no measurement rather than a poor one.
        self._min_height = min_height_m
        self._heights: dict[int, float] = {}
        self.counts: dict[str, int] = {}

    def count(self, key: str) -> None:
        self.counts[key] = self.counts.get(key, 0) + 1

    def ready_for(self, companion_t_us: int) -> bool:
        """Есть ли отсчеты автопилота ПОСЛЕ кадра (нужны для интерполяции)."""
        if not self.clock.valid:
            return False
        t = companion_t_us + self.clock.offset_us
        return self.attitude.latest_us >= t and self.distance.latest_us >= t

    def state_at_camera_time(self, companion_t_us: int):
        """(roll, pitch, yaw, height) на момент кадра или None."""
        if not self.clock.valid:
            return None
        return self._state_at(companion_t_us + self.clock.offset_us)

    def _state_at(self, autopilot_t_us: int):
        att = self.attitude.bracket(autopilot_t_us)
        rng = self.distance.bracket(autopilot_t_us)
        if att is None or rng is None:
            return None
        q = slerp(np.asarray(att[0]), np.asarray(att[1]), att[2])
        roll, pitch, yaw = quaternion_to_rpy(q)
        if cos(roll) * cos(pitch) < cos(self._max_tilt):
            return None
        distance = rng[0] + (rng[1] - rng[0]) * rng[2]
        # Дальномер жестко направлен вниз по оси корпуса: высота над
        # ровной землей — проекция на вертикаль.
        return roll, pitch, yaw, distance * cos(roll) * cos(pitch)

    def process(self, companion_t_us: int, gray: np.ndarray) -> BodyVelocityMeasurement | None:
        if not self.clock.valid:
            self.count("NO_TIME_SYNC")
            return None
        state = self._state_at(companion_t_us + self.clock.offset_us)
        if state is None:
            self.count("NO_ATTITUDE_OR_RANGE")
            self.vo.reject_frame(companion_t_us, "NO_ATTITUDE_OR_RANGE")
            return None
        roll, pitch, yaw, height = state
        if height < self._min_height:
            self.count("BELOW_MIN_HEIGHT")
            self.vo.reject_frame(companion_t_us, "BELOW_MIN_HEIGHT")
            return None
        result = self.vo.process(VisualFrame(companion_t_us, gray, height, yaw, roll, pitch))
        self._heights[companion_t_us] = height
        for t in [t for t in self._heights if t < companion_t_us - 2_000_000]:
            del self._heights[t]
        self.count(result.status)
        if not result.accepted:
            return None
        t0, t1 = result.interval_start_us, result.timestamp_us
        dt = (t1 - t0) * 1e-6
        h0 = self._heights.get(t0)
        if h0 is None or dt < self._min_dt or not isfinite(result.velocity_cov_nn):
            self.count("NO_RANGE_RATE")
            return None
        vd = -(height - h0) / dt
        var_vd = 2.0 * self._sigma_h ** 2 / dt ** 2
        mid = self._state_at((t0 + t1) // 2 + self.clock.offset_us)
        if mid is None:
            self.count("NO_ATTITUDE_OR_RANGE")
            return None
        r_nb = body_to_ned(mid[0], mid[1], mid[2])
        v_ned = np.array([result.vn_m_s, result.ve_m_s, vd])
        cov_ned = np.array([[result.velocity_cov_nn, result.velocity_cov_ne, 0.0],
                            [result.velocity_cov_ne, result.velocity_cov_ee, 0.0],
                            [0.0, 0.0, var_vd]])
        v_body = r_nb.T @ v_ned
        cov_body = r_nb.T @ cov_ned @ r_nb
        if not np.isfinite(v_body).all() or not np.isfinite(cov_body).all():
            self.count("NONFINITE")
            return None
        self.count("PUBLISHED")
        return BodyVelocityMeasurement((t0 + t1) // 2, tuple(float(v) for v in v_body),
                                       tuple(float(cov_body[i, i]) for i in range(3)), result, vd,
                                       float(mid[3]))
