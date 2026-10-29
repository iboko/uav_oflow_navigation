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

Options (3D swiss scene, offline replay of recorded frames,
tools/replay_vo_frames.py): with a surface model in the odometry each
feature gets its DSM depth (camera altitude = rangefinder height + DSM at
the position given with the frame); ``window_pairs`` > 1 sends the mean
velocity over that many contiguous pairs, stamped at the window midpoint.
The window cancels a ~2.5 Hz error from the frame-to-frame tilt
(EKF2 attitude and integrated gyro alike), which integrates to almost no
position error but dominates the per-pair velocity error. In the closed
loop the window's ~165 ms delay made the test flights swing at route
corners (docs/SWISS_SCENE_SITL_RU.md): keep 1.
"""
from __future__ import annotations

import bisect
from collections import deque
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
    rpy_mid: tuple[float, float, float] = (float("nan"),) * 3  # ориентация автопилота, середина
    # NED horizontal velocity of what is sent (the window mean with
    # window_pairs > 1); ``vo`` stays the last pair (anchor propagation).
    velocity_ne_sent: tuple[float, float] = (float("nan"),) * 2
    pairs_in_window: int = 1


class CompanionVo:
    def __init__(self, odometry: ContinuousPlanarOdometry, *, height_sigma_m: float = 0.02,
                 max_tilt_rad: float = 0.45, min_range_rate_dt_s: float = 0.02,
                 min_height_m: float = 0.5, window_pairs: int = 1,
                 vertical_gate: "VerticalVelocityGate | None" = None) -> None:
        if window_pairs < 1:
            raise ValueError("window_pairs must be >= 1")
        self.vo = odometry
        self._window_pairs = window_pairs
        self._window: deque = deque(maxlen=window_pairs)
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
        self._vertical_gate = vertical_gate
        self.last_vd_raw, self.last_vertical_rejected = float("nan"), False   # for the log
        self.counts: dict[str, int] = {}

    def count(self, key: str) -> None:
        self.counts[key] = self.counts.get(key, 0) + 1

    def ready_for(self, companion_t_us: int) -> bool:
        """Есть ли отсчеты автопилота ПОСЛЕ кадра (нужны для интерполяции)."""
        if not self.clock.valid:
            return False
        t = companion_t_us + self.clock.offset_us
        return self.attitude.latest_us >= t and self.distance.latest_us >= t

    def quaternion_at_camera_time(self, companion_t_us: int):
        """Autopilot attitude quaternion (FRD->NED) at the frame, or None."""
        if not self.clock.valid:
            return None
        att = self.attitude.bracket(companion_t_us + self.clock.offset_us)
        if att is None:
            return None
        return tuple(float(x) for x in slerp(np.asarray(att[0]), np.asarray(att[1]), att[2]))

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

    def process(self, companion_t_us: int, gray: np.ndarray,
                position_ne_m: tuple[float, float] | None = None) -> BodyVelocityMeasurement | None:
        """``position_ne_m``: estimated horizontal position (DSM lookup), if any."""
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
        ne, alt = None, None
        surface = self.vo.surface
        if surface is not None and position_ne_m is not None and all(isfinite(v) for v in position_ne_m):
            ne = (float(position_ne_m[0]), float(position_ne_m[1]))
            # The rangefinder (body down) hits the surface off the nadir when
            # tilted: sample the DSM at that point (50 m, 0.2 rad: 10 m away).
            down = body_to_ned(roll, pitch, yaw)[:, 2]
            slant = height / down[2]
            alt = height + float(surface.height_at(ne[0] + slant * down[0], ne[1] + slant * down[1])[0])
        result = self.vo.process(VisualFrame(companion_t_us, gray, height, yaw, roll, pitch, ne, alt))
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
        # Window of contiguous pairs (same segment, each starting where the
        # previous ended); a gap restarts it.
        if self._window and (self._window[-1][1] != t0 or self._window[-1][6] != result.segment_id):
            self._window.clear()
        self._window.append((t0, t1, result.vn_m_s, result.ve_m_s, vd,
                             (result.velocity_cov_nn, result.velocity_cov_ne, result.velocity_cov_ee),
                             result.segment_id))
        w0, w1 = self._window[0][0], self._window[-1][1]
        span = (w1 - w0) * 1e-6
        weights = [(e[1] - e[0]) * 1e-6 / span for e in self._window]
        vn, ve, vdw = (sum(wk * e[k] for wk, e in zip(weights, self._window)) for k in (2, 3, 4))
        # Model covariance of the mean: the per-pair mean, not divided by
        # the pair count (the errors of neighbouring pairs are correlated).
        cnn, cne, cee = (sum(wk * e[5][k] for wk, e in zip(weights, self._window)) for k in range(3))
        var_vd = 2.0 * self._sigma_h ** 2 / span ** 2
        self.last_vd_raw, self.last_vertical_rejected = float(vdw), False
        if self._vertical_gate is not None:
            t_mid = (w0 + w1) // 2
            if not self._vertical_gate.check(t_mid, vdw):
                self.last_vertical_rejected = True
                self.count("VO_VERTICAL_REJECTED")
                vdw, var_vd = self._vertical_gate.held(t_mid)
            var_vd = max(var_vd, self._vertical_gate.min_sigma ** 2)
        mid = self._state_at((w0 + w1) // 2 + self.clock.offset_us)
        if mid is None:
            self.count("NO_ATTITUDE_OR_RANGE")
            return None
        r_nb = body_to_ned(mid[0], mid[1], mid[2])
        v_ned = np.array([vn, ve, vdw])
        cov_ned = np.array([[cnn, cne, 0.0],
                            [cne, cee, 0.0],
                            [0.0, 0.0, var_vd]])
        v_body = r_nb.T @ v_ned
        cov_body = r_nb.T @ cov_ned @ r_nb
        if not np.isfinite(v_body).all() or not np.isfinite(cov_body).all():
            self.count("NONFINITE")
            return None
        self.count("PUBLISHED")
        return BodyVelocityMeasurement((w0 + w1) // 2, tuple(float(v) for v in v_body),
                                       tuple(float(cov_body[i, i]) for i in range(3)), result, float(vdw),
                                       float(mid[3]), (float(mid[0]), float(mid[1]), float(mid[2])),
                                       (float(vn), float(ve)), len(self._window))


class VelocityJumpGate:
    """Rejects VO velocities that jump more than the vehicle can accelerate.

    A multirotor tilted at most max_tilt accelerates horizontally at most
    g tan(max_tilt); between two VO measurements dt apart the velocity can
    change by that times dt, plus the VO noise of both. Series P/M/N (50 m,
    3D world): VO velocity error median 0.09-0.10 m/s, q95 0.28-0.32 m/s,
    but single samples 1-11 m/s off with hundreds of inliers. EKF2 takes
    VO with sigma ~1.6 m/s at that height, so a 1-2 m/s outlier passes its
    gate; in hover nothing corrects it and the position controller chases
    it (series P repeat, box_rf: 3.7 m/s and 26 deg tilt, 5.7 m error).

    Limits fixed from physics and the calibration flights (not the tests):
    max_accel = 1.5 g tan(0.25 rad) = 3.7 m/s^2 (VO itself drops frames
    beyond 0.25 rad), noise = 2 x q95 = 0.6 m/s. On calibration flights P:
    0.12 % of good samples (error < 0.5 m/s) rejected, 86 % of outliers
    (> 1 m/s); M/N: 0.10 % and 62 % (slow drifts are not jumps). After
    max_hold_s without an accepted sample the next one is taken as is.
    """

    def __init__(self, max_accel_m_s2: float = 3.7, noise_m_s: float = 0.6, max_hold_s: float = 1.0) -> None:
        self.max_accel = max_accel_m_s2
        self.noise = noise_m_s
        self.max_hold_s = max_hold_s
        self.last_t_us: int | None = None
        self.last_v: np.ndarray | None = None

    def check(self, t_us: int, v_ne) -> bool:
        v = np.asarray(v_ne, dtype=float)
        if not np.isfinite(v).all():
            return False
        if self.last_t_us is None or (t_us - self.last_t_us) * 1e-6 > self.max_hold_s:
            ok = True
        else:
            dt = max(0.0, (t_us - self.last_t_us) * 1e-6)
            ok = float(np.linalg.norm(v - self.last_v)) <= self.max_accel * dt + self.noise
        if ok:
            self.last_t_us, self.last_v = t_us, v
        return ok

    def held_velocity(self, t_us: int) -> tuple[tuple[float, float], float]:
        """Last accepted velocity and its variance per axis grown by the
        possible acceleration since: for the map anchor across a rejected
        interval (no gap, no invented precision)."""
        age = max(0.0, (t_us - self.last_t_us) * 1e-6)
        var = (self.max_accel * age) ** 2 + self.noise ** 2
        return (float(self.last_v[0]), float(self.last_v[1])), var


class VerticalVelocityGate:
    """Replaces implausible VO vertical velocities (range rate) by the last
    plausible one.

    The vertical velocity is the rangefinder height difference over a frame
    pair (66 ms). Over the 3D scene the beam jumps between roofs, trees and
    the ground, more so when tilted: a 2-m step is 30 m/s, sent with the
    rangefinder model sigma ~0.4 m/s. Calibration flights (series V without
    wind, gap-bridge flights with gusts): error median 0.09-0.15 m/s but q99
    3-22 m/s and up to 89 m/s. EKF2 rejects most, but after 1 s of failing
    EV velocity fusion it resets the whole velocity to the measurement: with
    strong gusts (seed 31) it took -22 m/s vertical, the controller cut the
    thrust and the vehicle fell from 50 m.

    Limits fixed from physics and the calibration flights (not the tests):
    a multirotor accelerates vertically by at most ~1 g (free fall down,
    thrust-to-weight ~2 up): max_accel = g; noise = 2 x q95 of the core error
    (robust sigma from the median, 0.18 m/s) = 0.7 m/s; |vd| <= 8 m/s, twice
    the q99 of the true vertical speed (3.0-3.8 m/s). A rejected value is
    replaced by the last accepted one, its variance grown by max_accel times
    the hold; after max_hold_s the next plausible value is taken as is, an
    implausible one is still replaced (by 0 if none was accepted).

    After a VO gap (nothing accepted for max_hold_s) the first value used
    to pass unchecked. EKF2 resets its velocity to the first EV velocity
    when the fusion restarts, and that value (a range difference over the
    first frame pair) was 4-8 m/s off: calibration flights C35 with
    injected dropouts, 7-16 vertical resets per flight, EKF2 off by up to
    7.9 m/s. With a reference (the autopilot's own vertical velocity, which
    the reset then leaves unchanged) the gate checks the value against it
    instead, and a value outside noise + max_accel * reference age is
    replaced by the reference until one agrees.

    What passes is still heavy-tailed (seven calibration flights: q95
    0.9-2.8 m/s, q99 3.1-6.6 m/s), so the variance sent has a floor
    min_sigma = max q99 / 2.58 = 2.5 m/s instead of the rangefinder model
    (~0.4 m/s): EKF2's EV velocity test then does not fail on the vertical
    and does not reset to it; the height stays on the barometer.

    The single last accepted value is a poor anchor: an outlier that passes
    (1-2 % of the accepted values are > 3 m/s off) makes the gate reject the
    correct values that follow and send the outlier instead. Series VG
    (6 strong-gust flights): 15 % of the replaced values were correct, 2.4 %
    of all sent values were correct values replaced by a worse one (added
    error q95 7.4 m/s), all in the check against the last accepted value.
    Option (robust=True, offline replay on the V1 calibration flights): the
    anchor is the median of the last anchor_n accepted values, and a value
    the anchor rejects is still accepted when it agrees with the autopilot
    reference within noise + max_accel * reference age.
    """

    def __init__(self, max_accel_m_s2: float = 9.81, noise_m_s: float = 0.7, max_abs_m_s: float = 8.0,
                 max_hold_s: float = 1.0, min_sigma_m_s: float = 2.5, robust: bool = False,
                 anchor_n: int = 5) -> None:
        if not all(isfinite(x) and x > 0 for x in (max_accel_m_s2, noise_m_s, max_abs_m_s, max_hold_s,
                                                   min_sigma_m_s)):
            raise ValueError("vertical gate limits must be finite and positive")
        if anchor_n < 1:
            raise ValueError("anchor_n must be >= 1")
        self.robust, self.anchor_n = bool(robust), int(anchor_n)
        self._accepted: list[float] = []
        self.max_accel, self.noise, self.max_abs, self.max_hold_s = max_accel_m_s2, noise_m_s, max_abs_m_s, max_hold_s
        self.min_sigma = min_sigma_m_s
        self.last_t_us: int | None = None
        self.last_vd = 0.0
        self.max_reference_age_s = 0.5
        self._ref: tuple[int, float] | None = None
        self._use_ref = False

    def set_reference(self, t_us: int, vd: float) -> None:
        """Autopilot vertical velocity at t_us (same clock as check())."""
        if isfinite(vd):
            self._ref = (int(t_us), float(vd))

    def _reference_at(self, t_us: int) -> tuple[float, float] | None:
        if self._ref is None:
            return None
        age = abs(t_us - self._ref[0]) * 1e-6
        return (self._ref[1], age) if age <= self.max_reference_age_s else None

    def check(self, t_us: int, vd: float) -> bool:
        self._use_ref = False
        if self.robust and self._accepted:
            self.last_vd = float(np.median(self._accepted))   # anchor, and what held() sends
        expired = self.last_t_us is None or (t_us - self.last_t_us) * 1e-6 > self.max_hold_s
        ref = self._reference_at(t_us) if expired else None
        if not isfinite(vd) or abs(vd) > self.max_abs:
            self._use_ref = ref is not None
            return False
        if expired and ref is not None:
            ok = abs(vd - ref[0]) <= self.noise + self.max_accel * ref[1]
            self._use_ref = not ok
        elif expired:
            ok = True
        else:
            dt = max(0.0, (t_us - self.last_t_us) * 1e-6)
            ok = abs(vd - self.last_vd) <= self.max_accel * dt + self.noise
            if not ok and self.robust:
                ref = self._reference_at(t_us)
                ok = ref is not None and abs(vd - ref[0]) <= self.noise + self.max_accel * ref[1]
        if ok:
            self.last_t_us, self.last_vd = t_us, float(vd)
            if self.robust:
                self._accepted = (self._accepted + [float(vd)])[-self.anchor_n:]
        return ok

    def held(self, t_us: int) -> tuple[float, float]:
        """Replacement for a rejected value at t_us and its variance: the
        reference after a gap, else the last accepted value (the anchor
        median with robust=True)."""
        ref = self._reference_at(t_us) if self._use_ref else None
        if ref is not None:
            return ref[0], min((self.max_accel * ref[1]) ** 2 + self.noise ** 2, self.max_abs ** 2)
        if self.last_t_us is None:
            return 0.0, self.max_abs ** 2
        age = max(0.0, (t_us - self.last_t_us) * 1e-6)
        return self.last_vd, min((self.max_accel * age) ** 2 + self.noise ** 2, self.max_abs ** 2)
