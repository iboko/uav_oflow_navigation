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
* сопровождение: NIS привязки относительно распространенного положения
  не больше χ²(2; 0,99). Отклоненная привязка не меняет положение, и
  захват заново по самим привязкам при этом не начинается: ложная, но
  самосогласованная карта (сдвиг георефа) иначе была бы принята.
  Захват заново нужен только после потери (СКО положения > max_sigma_m
  без принятых привязок) — тогда сдвинутая карта будет принята:
  внешней проверки, кроме VO, у компаньона нет.

Курс: VO переводит скорость в N/E по курсу автопилота (магнитометр,
ошибка 5-10°). Привязка к карте измеряет курс; средняя поправка по
принятым привязкам поворачивает перемещения VO, пока она не набрана —
неопределенность курса входит в ковариацию.

Неучтенное: ошибки VO и привязки по одним кадрам коррелированы, а EKF2
получает и это положение, и скорость VO (двойной учет VO). Ковариация
положения при распространении растет по модели ошибок VO.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot, isfinite

import numpy as np

CHI2_2D_99 = 9.210340371976184


@dataclass
class AnchorStatus:
    locked: bool
    position_ne_m: tuple[float, float] | None
    covariance_ne_m2: tuple[tuple[float, float], tuple[float, float]] | None
    counts: dict = field(default_factory=dict)
    heading_offset_rad: float = 0.0


class MapAnchor:
    def __init__(self, *, bias_ne_m: tuple[float, float],
                 covariance_ne_m2: tuple[tuple[float, float], tuple[float, float]],
                 vo_scale_sigma: float = 0.05, acquisition_fixes: int = 3,
                 acquisition_tolerance_m: float = 0.3, max_sigma_m: float = 3.0,
                 nis_gate: float = CHI2_2D_99, max_gap_speed_m_s: float = 2.0,
                 heading_sigma_unknown_rad: float = 0.15, heading_sigma_known_rad: float = 0.03,
                 heading_fixes_needed: int = 3) -> None:
        r = np.asarray(covariance_ne_m2, dtype=float)
        b = np.asarray(bias_ne_m, dtype=float)
        if r.shape != (2, 2) or b.shape != (2,) or not np.isfinite(r).all() or not np.isfinite(b).all():
            raise ValueError("Нужны конечные смещение и ковариация ошибки карты")
        if not np.allclose(r, r.T) or np.min(np.linalg.eigvalsh(r)) <= 0:
            raise ValueError("Ковариация ошибки карты не положительно определена")
        if not (0 <= vo_scale_sigma < 0.5 and acquisition_fixes >= 2 and acquisition_tolerance_m > 0
                and max_sigma_m > 0 and nis_gate > 0):
            raise ValueError("Недопустимые параметры привязки")
        self._r = r
        self._bias = b
        self._scale_sigma = vo_scale_sigma
        self._n_acq = acquisition_fixes
        self._acq_tol = acquisition_tolerance_m
        self._max_var = max_sigma_m ** 2
        self._gate = nis_gate
        self._gap_speed = max_gap_speed_m_s
        # Heading of the autopilot (used by VO for N/E) relative to the map
        # frame, measured by the map fixes: magnetometer headings are off by
        # 5-10 deg, which rotates every VO displacement.
        self._sigma_psi_unknown = heading_sigma_unknown_rad
        self._sigma_psi_known = heading_sigma_known_rad
        self._psi_needed = heading_fixes_needed
        self._psi_sum = np.zeros(2)  # sum of (cos, sin) of measured offsets
        self._psi_n = 0
        self._p: np.ndarray | None = None
        self._cov: np.ndarray | None = None
        self._t_us = 0
        self._candidates: list[tuple[int, np.ndarray]] = []
        self._vo_since_candidate = np.zeros(2)
        self.counts: dict[str, int] = {}
        self.last_nis = float("nan")

    def _count(self, key: str) -> None:
        self.counts[key] = self.counts.get(key, 0) + 1

    @property
    def locked(self) -> bool:
        return self._p is not None

    def propagate(self, t0_us: int, t1_us: int, v_ne: tuple[float, float],
                  cov_ne: tuple[tuple[float, float], tuple[float, float]]) -> None:
        """VO-перемещение за [t0, t1]; интервалы идут подряд по времени."""
        dt = (t1_us - t0_us) * 1e-6
        v = np.asarray(v_ne, dtype=float)
        c = np.asarray(cov_ne, dtype=float)
        if dt <= 0 or not np.isfinite(v).all() or not np.isfinite(c).all():
            self._count("BAD_VO_INTERVAL")
            return
        psi = self.heading_offset_rad
        c_psi, s_psi = np.cos(psi), np.sin(psi)
        rot = np.array([[c_psi, -s_psi], [s_psi, c_psi]])
        d = rot @ (v * dt)
        c = rot @ c @ rot.T
        sigma_psi = self._sigma_psi_known if self._psi_n >= self._psi_needed else self._sigma_psi_unknown
        gap_s = (t0_us - self._t_us) * 1e-6 if self._t_us else 0.0
        if gap_s < -1e-3:
            self._count("BAD_VO_INTERVAL")
            return
        if self._p is None:
            if self._candidates and gap_s > 1e-3:
                # Перемещение за разрыв VO неизвестно: захват заново.
                self._candidates.clear()
                self._count("ACQUISITION_VO_GAP")
            if self._candidates:
                self._vo_since_candidate += d
            self._t_us = t1_us
            return
        self._p = self._p + d
        # Шум VO по паре + систематическая ошибка масштаба (доля пути);
        # за разрыв VO — неизвестное перемещение со скоростью до max_gap_speed.
        self._cov = (self._cov + c * dt * dt
                     + np.eye(2) * (self._scale_sigma * float(np.linalg.norm(d))) ** 2
                     + np.eye(2) * (sigma_psi * float(np.linalg.norm(d))) ** 2
                     + np.eye(2) * (self._gap_speed * max(gap_s, 0.0)) ** 2)
        if gap_s > 1e-3:
            self._count("VO_GAP_INFLATED")
        self._t_us = t1_us
        if np.max(np.linalg.eigvalsh(self._cov)) > self._max_var:
            # Без привязок слишком долго: положение больше не передается.
            self._p = None
            self._cov = None
            self._candidates.clear()
            self._count("LOCK_LOST_UNCERTAINTY")
            # Повторный захват — только по кинематически согласованным
            # привязкам (см. описание модуля): компаньон их снова наберет.

    @property
    def heading_offset_rad(self) -> float:
        """Map heading minus autopilot heading; 0 until measured."""
        if self._psi_n < self._psi_needed:
            return 0.0
        return float(np.arctan2(self._psi_sum[1], self._psi_sum[0]))

    def _heading(self, offset_rad: float) -> None:
        if isfinite(offset_rad):
            # Exponential window (~20 fixes): the magnetometer error changes slowly.
            self._psi_sum = 0.95 * self._psi_sum + (np.cos(offset_rad), np.sin(offset_rad))
            self._psi_n += 1

    def fix(self, t_us: int, accepted: bool, north_m: float, east_m: float,
            heading_offset_rad: float = float("nan")) -> str:
        """Результат привязки на момент кадра t_us (после propagate до t_us).

        heading_offset_rad: курс по карте минус курс автопилота для этого кадра.
        Учитывается только у привязок, принятых проверками (захват, NIS).
        """
        if not accepted or not (isfinite(north_m) and isfinite(east_m)):
            self._count("FIX_NOT_ACCEPTED_BY_LOCALIZER")
            return "FIX_NOT_ACCEPTED_BY_LOCALIZER"
        z = np.array([north_m, east_m]) - self._bias
        if t_us != self._t_us:
            # Положение известно только на конец последнего интервала VO.
            self._count("FIX_TIME_MISMATCH")
            return "FIX_TIME_MISMATCH"
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
        k = self._cov @ np.linalg.inv(s)
        self._p = self._p + k @ y
        self._cov = (np.eye(2) - k) @ self._cov
        self._cov = 0.5 * (self._cov + self._cov.T)
        self._heading(heading_offset_rad)
        self._count("FIX_FUSED")
        return "FIX_FUSED"

    def _acquire(self, t_us: int, z: np.ndarray) -> str:
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
        self._candidates.clear()
        self._count("LOCK_ACQUIRED")
        return "LOCK_ACQUIRED"

    def status(self) -> AnchorStatus:
        if self._p is None:
            return AnchorStatus(False, None, None, dict(self.counts), self.heading_offset_rad)
        return AnchorStatus(True, (float(self._p[0]), float(self._p[1])),
                            tuple(tuple(float(v) for v in row) for row in self._cov), dict(self.counts),
                            self.heading_offset_rad)
