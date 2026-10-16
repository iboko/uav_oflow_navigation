"""Бортовая привязка к ортофотоплану на компаньоне: положение для EKF2.

Положение = последняя принятая привязка к карте + перемещение по VO
после нее. Так положение есть в каждом сообщении ODOMETRY: EKF2
останавливает слияние внешнего положения на каждом сообщении без него
(и при новом старте без ГНСС сбрасывает положение).

Привязка принимается только с откалиброванной ковариацией ошибки карты
(`MapErrorCalibration`, отдельные участки) и только если согласуется с
текущим положением:

* захват: три подряд привязки, разности которых совпадают с перемещением
  по VO (кинематическая проверка, не только по самой карте);
* сопровождение: NIS не больше χ²(2; 0,99) и изменение привязки
  согласуется с накопленным перемещением VO. После этих проверок
  пересечение ковариаций (CI) учитывает неизвестную корреляцию карты
  и прогноза. Отклоненная привязка не меняет положение, и
  захват заново по самим привязкам при этом не начинается: ложная, но
  самосогласованная карта (сдвиг георефа) иначе была бы принята.
  Захват заново нужен только после потери (СКО положения > max_sigma_m
  без принятых привязок) — тогда сдвинутая карта будет принята:
  внешней проверки, кроме VO, у компаньона нет.

Курс: VO переводит скорость в N/E по курсу автопилота (магнитометр,
ошибка 5-10°). Привязка к карте измеряет курс; средняя поправка по
принятым привязкам поворачивает перемещения VO, пока она не набрана —
неопределенность курса входит в ковариацию.

С курсом по карте в EKF2 (EV yaw) курс автопилота сам поворачивается к
отправленному: поправка «карта минус автопилот», прибавленная к новому
курсу автопилота, повторяется (обратная связь; тест M: курс ушел на
48°). Тогда поправка берется относительно курса по гироскопу
(`GyroHeading`), на который EKF2 не влияет, а `propagate` получает
разность «гироскоп минус автопилот».

CI не исправляет неверную калибровку и не исключает двойной учет VO:
EKF2 получает и это положение, и скорость VO. Ковариация положения
при распространении растет по модели ошибок VO. Проверка скачка
имеет инженерный порог и не является гарантией целостности карты.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot, isfinite

import numpy as np

from .map_covariance_intersection import _covariance, covariance_intersection_2d

CHI2_2D_99 = 9.210340371976184


@dataclass
class AnchorStatus:
    locked: bool
    position_ne_m: tuple[float, float] | None
    covariance_ne_m2: tuple[tuple[float, float], tuple[float, float]] | None
    counts: dict = field(default_factory=dict)
    heading_offset_rad: float = 0.0


@dataclass(frozen=True)
class AnchorMeasurement:
    timestamp_us: int
    position_ne_m: tuple[float, float]
    covariance_ne_m2: tuple[tuple[float, float], tuple[float, float]]

    def mavlink_pose_covariance(self, height_variance_m2: float) -> tuple[float, ...]:
        """Upper triangle of [N,E,D,roll,pitch,yaw], including the NE term."""
        if not isfinite(height_variance_m2) or height_variance_m2 < 0:
            raise ValueError("Некорректная дисперсия высоты")
        p = _covariance(self.covariance_ne_m2, "положение для MAVLink")
        values = [0.0] * 21
        values[0], values[1], values[6], values[11] = p[0, 0], p[0, 1], p[1, 1], height_variance_m2
        values[15] = values[18] = values[20] = float("nan")
        return tuple(float(x) for x in values)


class MapAnchor:
    def __init__(self, *, bias_ne_m: tuple[float, float],
                 covariance_ne_m2: tuple[tuple[float, float], tuple[float, float]],
                 vo_scale_sigma: float = 0.05, acquisition_fixes: int = 3,
                 acquisition_tolerance_m: float = 0.3, max_sigma_m: float = 3.0,
                 nis_gate: float = CHI2_2D_99, max_gap_speed_m_s: float = 2.0,
                 heading_sigma_unknown_rad: float = 0.15, heading_sigma_known_rad: float = 0.03,
                 heading_fixes_needed: int = 3, max_map_step_error_m: float = 1.0) -> None:
        r = np.asarray(covariance_ne_m2, dtype=float)
        b = np.asarray(bias_ne_m, dtype=float)
        if r.shape != (2, 2) or b.shape != (2,) or not np.isfinite(r).all() or not np.isfinite(b).all():
            raise ValueError("Нужны конечные смещение и ковариация ошибки карты")
        r = _covariance(r, "ошибка карты")
        if not all(isfinite(x) for x in (vo_scale_sigma, acquisition_tolerance_m, max_sigma_m,
                nis_gate, max_gap_speed_m_s, heading_sigma_unknown_rad, heading_sigma_known_rad,
                max_map_step_error_m)):
            raise ValueError("Неконечные параметры привязки")
        if not (0 <= vo_scale_sigma < 0.5 and acquisition_fixes >= 2 and acquisition_tolerance_m > 0
                and max_sigma_m > 0 and nis_gate > 0 and max_gap_speed_m_s >= 0
                and heading_sigma_unknown_rad >= 0 and heading_sigma_known_rad >= 0
                and heading_fixes_needed >= 1 and max_map_step_error_m > 0):
            raise ValueError("Недопустимые параметры привязки")
        self._r = r
        self._bias = b
        self._scale_sigma = vo_scale_sigma
        self._n_acq = acquisition_fixes
        self._acq_tol = acquisition_tolerance_m
        self._max_var = max_sigma_m ** 2
        self._gate = nis_gate
        self._gap_speed = max_gap_speed_m_s
        self._step_limit = max_map_step_error_m
        self._last_map_z: np.ndarray | None = None
        self._vo_since_fix = np.zeros(2)
        self._step_budget_m = 0.0
        self._drift_sigma_m = np.zeros(3)  # scale, heading, unobserved VO gaps
        self._prior_p: np.ndarray | None = None
        self._prior_cov: np.ndarray | None = None
        self._interval_start_us = 0
        self._interval_velocity_ne = np.zeros(2)
        self._interval_velocity_cov = np.eye(2)
        self._interval_heading_sigma = 0.0
        # Heading of the autopilot (used by VO for N/E) relative to the map
        # frame, measured by the map fixes: magnetometer headings are off by
        # 5-10 deg, which rotates every VO displacement.
        self._sigma_psi_unknown = heading_sigma_unknown_rad
        self._sigma_psi_known = heading_sigma_known_rad
        self._psi_needed = heading_fixes_needed
        self._psi_sum = np.zeros(2)  # sum of (cos, sin) of measured offsets
        self._psi_n = 0
        self._psi_var = 0.0          # exponentially weighted spread of the offsets
        self._p: np.ndarray | None = None
        self._cov: np.ndarray | None = None
        self._t_us = 0
        self._candidates: list[tuple[int, np.ndarray]] = []
        self._vo_since_candidate = np.zeros(2)
        self.counts: dict[str, int] = {}
        self.last_nis = float("nan")
        self.last_step_error_m = float("nan")
        self.last_ci_weight_prior = float("nan")
        self.last_fix_us = 0   # time of the last fused (or locking) map fix

    def _count(self, key: str) -> None:
        self.counts[key] = self.counts.get(key, 0) + 1

    @property
    def locked(self) -> bool:
        return self._p is not None

    def propagate(self, t0_us: int, t1_us: int, v_ne: tuple[float, float],
                  cov_ne: tuple[tuple[float, float], tuple[float, float]],
                  reference_minus_autopilot_rad: float = 0.0) -> None:
        """VO-перемещение за [t0, t1]; интервалы идут подряд по времени.

        reference_minus_autopilot_rad: курс, относительно которого измерена
        поправка (например, по гироскопу), минус курс автопилота, по
        которому VO перевела скорость в N/E; 0 — поправка от автопилота."""
        dt = (t1_us - t0_us) * 1e-6
        v = np.asarray(v_ne, dtype=float)
        c = np.asarray(cov_ne, dtype=float)
        if not isfinite(dt) or dt <= 0 or v.shape != (2,) or not np.isfinite(v).all():
            self._count("BAD_VO_INTERVAL")
            return
        try:
            c = _covariance(c, "скорость VO")
        except ValueError:
            self._count("BAD_VO_COVARIANCE")
            return
        psi = self.heading_offset_rad
        if self.heading_known and isfinite(reference_minus_autopilot_rad):
            # Until measured the offset is 0 against the autopilot heading itself.
            psi += reference_minus_autopilot_rad
        c_psi, s_psi = np.cos(psi), np.sin(psi)
        rot = np.array([[c_psi, -s_psi], [s_psi, c_psi]])
        d = rot @ (v * dt)
        c = rot @ c @ rot.T
        sigma_psi = self._sigma_psi_known if self._psi_n >= self._psi_needed else self._sigma_psi_unknown
        gap_s = (t0_us - self._t_us) * 1e-6 if self._t_us else 0.0
        if gap_s < -1e-3:
            self._count("BAD_VO_INTERVAL")
            return
        self._interval_start_us = t0_us
        self._interval_velocity_ne = rot @ v
        self._interval_velocity_cov = c.copy()
        self._interval_heading_sigma = sigma_psi
        if self._p is None:
            if self._prior_p is not None:
                # Dead reckoning of the acquisition prior: VO noise, the
                # path-proportional scale/heading error (conservatively as
                # independent increments) and unknown motion over VO gaps.
                dist = float(np.linalg.norm(d))
                self._prior_p = self._prior_p + d
                self._prior_cov = (self._prior_cov + c * dt * dt
                                   + np.eye(2) * ((self._scale_sigma + sigma_psi) * dist) ** 2
                                   + np.eye(2) * (self._gap_speed * max(gap_s, 0.0)) ** 2)
            if self._candidates and gap_s > 1e-3:
                # Перемещение за разрыв VO неизвестно: захват заново.
                self._candidates.clear()
                self._count("ACQUISITION_VO_GAP")
            if self._candidates:
                self._vo_since_candidate += d
            self._t_us = t1_us
            return
        self._p = self._p + d
        self._vo_since_fix += d
        self._step_budget_m += ((self._scale_sigma + sigma_psi) * float(np.linalg.norm(d))
                               + self._gap_speed * max(gap_s, 0.0))
        # Persistent scale/heading errors accumulate with total path, not
        # with sqrt(number of frames). Add only the increment of the squared
        # accumulated budgets; the same displacement split into more frames
        # must not make its systematic uncertainty disappear.
        old_drift_var = float(self._drift_sigma_m @ self._drift_sigma_m)
        distance = float(np.linalg.norm(d))
        self._drift_sigma_m += np.array([self._scale_sigma * distance, sigma_psi * distance,
                                        self._gap_speed * max(gap_s, 0.0)])
        drift_var = float(self._drift_sigma_m @ self._drift_sigma_m)
        self._cov = self._cov + c * dt * dt + np.eye(2) * (drift_var - old_drift_var)
        if gap_s > 1e-3:
            self._count("VO_GAP_INFLATED")
        self._t_us = t1_us
        if np.max(np.linalg.eigvalsh(self._cov)) > self._max_var:
            # Без привязок слишком долго: положение больше не передается.
            # Оценка остается априорной для повторного захвата: ложное место
            # карты (серия H: место сдвинуто на 21 м, привязки
            # внутри места согласованы) иначе захватывалось бы заново.
            self._prior_p, self._prior_cov = self._p.copy(), self._cov.copy()
            self._p = None
            self._cov = None
            self._last_map_z = None
            self._candidates.clear()
            self._count("LOCK_LOST_UNCERTAINTY")
            # Повторный захват — только по кинематически согласованным
            # привязкам (см. описание модуля): компаньон их снова наберет.

    @property
    def heading_known(self) -> bool:
        return self._psi_n >= self._psi_needed

    def heading_sigma_rad(self, floor_rad: float = 0.05) -> float:
        """Spread of the measured offsets, never below floor_rad: the same
        offset is repeated in many autopilot messages (not independent)."""
        return max(floor_rad, float(np.sqrt(self._psi_var)))

    @property
    def heading_offset_rad(self) -> float:
        """Map heading minus autopilot heading; 0 until measured."""
        if self._psi_n < self._psi_needed:
            return 0.0
        return float(np.arctan2(self._psi_sum[1], self._psi_sum[0]))

    def _heading(self, offset_rad: float) -> None:
        if isfinite(offset_rad):
            if self._psi_n >= self._psi_needed:
                d = float(np.arctan2(np.sin(offset_rad - self.heading_offset_rad),
                                     np.cos(offset_rad - self.heading_offset_rad)))
                self._psi_var = 0.95 * self._psi_var + 0.05 * d * d
            # Exponential window (~20 fixes): the magnetometer error changes slowly.
            self._psi_sum = 0.95 * self._psi_sum + (np.cos(offset_rad), np.sin(offset_rad))
            self._psi_n += 1

    def fix(self, t_us: int, accepted: bool, north_m: float, east_m: float,
            heading_offset_rad: float = float("nan"),
            covariance_ne_m2: tuple[tuple[float, float], tuple[float, float]] | None = None) -> str:
        """Результат привязки на момент кадра t_us (после propagate до t_us).

        heading_offset_rad: курс по карте минус курс автопилота для этого кадра.
        Учитывается только у привязок, принятых проверками (захват, NIS).
        covariance_ne_m2: ковариация этой привязки (например, пересчитанная
        калибровочная на высоту кадра); по умолчанию — заданная при создании.
        """
        if not accepted or not (isfinite(north_m) and isfinite(east_m)):
            self._count("FIX_NOT_ACCEPTED_BY_LOCALIZER")
            return "FIX_NOT_ACCEPTED_BY_LOCALIZER"
        if covariance_ne_m2 is not None:
            try:
                self._r = _covariance(np.asarray(covariance_ne_m2, dtype=float), "ковариация привязки")
            except ValueError:
                self._count("FIX_BAD_COVARIANCE")
                return "FIX_BAD_COVARIANCE"
        z = np.array([north_m, east_m]) - self._bias
        if t_us != self._t_us:
            # Положение известно только на конец последнего интервала VO.
            self._count("FIX_TIME_MISMATCH")
            return "FIX_TIME_MISMATCH"
        if self._p is not None and t_us <= self.last_fix_us:
            self._count("FIX_DUPLICATE_OR_OLD")
            return "FIX_DUPLICATE_OR_OLD"
        if self._p is None:
            result = self._acquire(t_us, z)
            if result == "LOCK_ACQUIRED":
                self._heading(heading_offset_rad)
            return result
        y = z - self._p
        s = self._cov + self._r
        nis = float(y @ np.linalg.solve(s, y))
        self.last_nis = nis
        if not isfinite(nis) or nis > self._gate:
            self._count("FIX_REJECTED_NIS")
            return "FIX_REJECTED_NIS"
        # A wide marginal covariance must not hide a sudden georeference
        # jump. Compare changes since the last accepted map fix with VO,
        # retaining the same reference after every rejection. This is a
        # configurable engineering bound, not a statistical integrity proof.
        self.last_step_error_m = float(np.linalg.norm(z - self._last_map_z - self._vo_since_fix))
        if self.last_step_error_m > self._step_limit + self._step_budget_m:
            self._count("FIX_REJECTED_STEP")
            return "FIX_REJECTED_STEP"
        self._p, self._cov, self.last_ci_weight_prior = covariance_intersection_2d(
            self._p, self._cov, z, self._r)
        self._last_map_z = z.copy()
        self._vo_since_fix = np.zeros(2)
        self._step_budget_m = 0.0
        self._drift_sigma_m[:] = 0.0
        self._heading(heading_offset_rad)
        self.last_fix_us = t_us
        self._count("FIX_FUSED")
        return "FIX_FUSED"

    def set_acquisition_prior(self, position_ne_m, covariance_ne_m2) -> None:
        """Independent prior for the next acquisition (e.g. the GNSS-aided
        autopilot position before the first lock). Ignored while locked."""
        if self._p is not None:
            return
        p = np.asarray(position_ne_m, dtype=float)
        if p.shape != (2,) or not np.isfinite(p).all():
            return
        self._prior_p = p.copy()
        self._prior_cov = _covariance(np.asarray(covariance_ne_m2, dtype=float), "априорное положение")

    def _acquire(self, t_us: int, z: np.ndarray) -> str:
        if self._prior_p is not None:
            y = z - self._prior_p
            nis = float(y @ np.linalg.solve(self._prior_cov + self._r, y))
            if not nis <= self._gate:
                self._candidates.clear()
                self._count("ACQUISITION_REJECTED_PRIOR")
                return "ACQUISITION_REJECTED_PRIOR"
        if self._candidates:
            t_prev, z_prev = self._candidates[-1]
            moved_vo = self._vo_since_candidate
            moved_map = z - z_prev
            tol = self._acq_tol + (self._scale_sigma + self._sigma_psi_unknown) * float(np.linalg.norm(moved_vo))
            if hypot(*(moved_map - moved_vo)) > tol or t_us <= t_prev:
                # Несогласованная пара: захват начинается заново с этой привязки.
                self._candidates = [(t_us, z)]
                self._vo_since_candidate = np.zeros(2)
                self._count("ACQUISITION_INCONSISTENT")
                return "ACQUISITION_INCONSISTENT"
        self._candidates.append((t_us, z))
        self._vo_since_candidate = np.zeros(2)
        if len(self._candidates) < self._n_acq:
            self._count("ACQUISITION_PENDING")
            return "ACQUISITION_PENDING"
        self._p = z.copy()
        self._cov = self._r.copy()
        self._last_map_z = z.copy()
        self._vo_since_fix = np.zeros(2)
        self._step_budget_m = 0.0
        self._drift_sigma_m[:] = 0.0
        self._candidates.clear()
        self._prior_p = self._prior_cov = None
        self.last_fix_us = t_us
        self._count("LOCK_ACQUIRED")
        return "LOCK_ACQUIRED"

    def status(self) -> AnchorStatus:
        if self._p is None:
            return AnchorStatus(False, None, None, dict(self.counts), self.heading_offset_rad)
        return AnchorStatus(True, (float(self._p[0]), float(self._p[1])),
                            tuple(tuple(float(v) for v in row) for row in self._cov), dict(self.counts),
                            self.heading_offset_rad)

    def measurement_at(self, timestamp_us: int, *, max_fix_age_s: float) -> AnchorMeasurement | None:
        """Map-backed position within the most recent VO interval only.

        A midpoint VO message must use the same heading-corrected velocity
        used by propagate(). No extrapolation outside the stored interval.
        End-state covariance is retained and inflated for the backward
        displacement with an unknown-cross-correlation bound; it is never
        reduced by pretending this velocity is an independent observation.
        """
        if not isfinite(max_fix_age_s) or max_fix_age_s <= 0:
            raise ValueError("Возраст привязки должен быть конечным и положительным")
        if (self._p is None or not isfinite(timestamp_us)
                or not self._interval_start_us <= timestamp_us <= self._t_us
                or not 0 <= self._t_us - self.last_fix_us <= max_fix_age_s * 1e6):
            return None
        back_s = (self._t_us - timestamp_us) * 1e-6
        back = self._interval_velocity_ne * back_s
        cov = self._cov.copy()
        if back_s > 0:
            q = self._interval_velocity_cov * back_s ** 2 + np.eye(2) * (
                self._scale_sigma ** 2 + self._interval_heading_sigma ** 2) * float(back @ back)
            # ||C_p,d + C_d,p|| <= 2 sqrt(lambda_max(P) lambda_max(Q)).
            cross_bound = 2 * np.sqrt(np.linalg.eigvalsh(cov).max() * np.linalg.eigvalsh(q).max())
            cov += q + np.eye(2) * cross_bound
        position = self._p - back
        return AnchorMeasurement(int(timestamp_us), tuple(float(x) for x in position),
                                 tuple(tuple(float(x) for x in row) for row in cov))


class GyroHeading:
    """Курс, проинтегрированный по угловым скоростям автопилота.

    Скорости (ATTITUDE_QUATERNION: rollspeed/pitchspeed/yawspeed, FRD)
    не меняются, когда EKF2 исправляет курс (слияние EV yaw, сброс по
    магнитометру), поэтому поправка курса по карте относительно этого
    курса не участвует в обратной связи. Уходит со смещением гироскопа
    (медленно; поправка по карте — скользящее среднее). Начало —
    курс автопилота в первом отсчете.

    ψ' = (q sin φ + r cos φ) / cos θ; трапеции между отсчетами;
    в разрыве больше max_gap_us — скорость последнего отсчета (счетчик gaps).
    """

    def __init__(self, max_gap_us: int = 200_000, keep_us: int = 5_000_000) -> None:
        self._t: list[int] = []
        self._psi: list[float] = []
        self._last_rate = 0.0
        self._max_gap = max_gap_us
        self._keep = keep_us
        self.gaps = 0

    @staticmethod
    def _rate(roll: float, pitch: float, rates_frd) -> float:
        _, q, r = rates_frd
        return (q * np.sin(roll) + r * np.cos(roll)) / max(np.cos(pitch), 0.1)

    def add(self, t_us: int, roll: float, pitch: float, yaw: float, rates_frd) -> None:
        if not all(map(isfinite, (roll, pitch, yaw, *rates_frd))):
            return
        rate = float(self._rate(roll, pitch, rates_frd))
        if not self._t:
            self._t.append(t_us)
            self._psi.append(float(yaw))
            self._last_rate = rate
            return
        dt_us = t_us - self._t[-1]
        if dt_us <= 0:
            return
        if dt_us > self._max_gap:
            self.gaps += 1
            step = self._last_rate * dt_us * 1e-6
        else:
            step = 0.5 * (self._last_rate + rate) * dt_us * 1e-6
        self._t.append(t_us)
        self._psi.append(self._psi[-1] + step)
        self._last_rate = rate
        while self._t[0] < t_us - self._keep:
            self._t.pop(0)
            self._psi.pop(0)

    def at(self, t_us: int) -> float:
        """Курс в момент t_us (часы автопилота), без экстраполяции; NaN вне истории."""
        if not self._t or t_us < self._t[0] or t_us > self._t[-1]:
            return float("nan")
        return float(np.interp(t_us, self._t, self._psi))
