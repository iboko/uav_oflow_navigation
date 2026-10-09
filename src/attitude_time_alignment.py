"""Согласование времени изображений с журналом ориентации ИНС/автопилота.

Ориентация задается кватернионом СК БВС FRD -> NED [w,x,y,z].
Это НЕ интегрирование сырых измерений гироскопов и НЕ VIO.
Никакой экстраполяции, только SLERP между двумя надежными отсчетами.
"""
from __future__ import annotations

import bisect
import csv
from dataclasses import dataclass
from math import asin, atan2, copysign, isfinite, pi, sin, sqrt

import numpy as np


@dataclass(frozen=True)
class AttitudeRecord:
    timestamp_us: int
    qw: float
    qx: float
    qy: float
    qz: float


def _quaternion(record: AttitudeRecord) -> np.ndarray:
    q = np.array([record.qw, record.qx, record.qy, record.qz], dtype=np.float64)
    if not np.isfinite(q).all():
        raise ValueError("Кватернион содержит недопустимые компоненты")
    norm = float(np.linalg.norm(q))
    if norm < 1e-8 or not 0.8 <= norm <= 1.2:
        raise ValueError("Кватернион не нормирован")
    return q / norm


def quaternion_to_rpy(q: np.ndarray) -> tuple[float, float, float]:
    """Конвенция PX4 FRD->NED, Rz(yaw) Ry(pitch) Rx(roll)."""
    w, x, y, z = [float(t) for t in q]
    roll = atan2(2.0 * (w * x + y * z),
                 1.0 - 2.0 * (x * x + y * y))
    sp = 2.0 * (w * y - z * x)
    pitch = copysign(pi / 2.0, sp) if abs(sp) >= 1.0 else asin(sp)
    yaw = atan2(2.0 * (w * z + x * y),
                1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("Интерполяция требует 0 <= alpha <= 1")
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(q0 @ q1)
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        out = q0 + alpha * (q1 - q0)
        return out / np.linalg.norm(out)
    omega = float(np.arccos(dot))
    sn = sin(omega)
    out = sin((1.0 - alpha) * omega) / sn * q0 + sin(alpha * omega) / sn * q1
    return out / np.linalg.norm(out)


class AttitudeTimeline:
    """Бортовая оценка ориентации, интерполированная к середине экспозиции.

    camera_to_attitude_offset_us: t_оценки_ориентации = t_камеры + offset
    (включает уже откалиброванную разницу шкал времени).
    """

    def __init__(
        self,
        records: list[AttitudeRecord],
        *,
        max_sample_gap_us: int = 25000,
        camera_to_attitude_offset_us: int = 0,
    ) -> None:
        if not records or max_sample_gap_us <= 0:
            raise ValueError("Нет ориентации или неверный максимальный интервал")
        if not isinstance(camera_to_attitude_offset_us, int):
            raise ValueError("Смещение временных шкал должно задаваться в мкс")
        times = [rec.timestamp_us for rec in records]
        if any(not isinstance(t, int) or t <= 0 for t in times):
            raise ValueError("Неверные временные метки ориентации")
        if any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError("Метки ориентации должны строго возрастать")
        self._quats = [_quaternion(rec) for rec in records]
        self._times = times
        self._max_gap = max_sample_gap_us
        self._offset = camera_to_attitude_offset_us

    @classmethod
    def from_csv(cls, path: str, **kwargs) -> "AttitudeTimeline":
        records = []
        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            required = {"timestamp_us", "qw", "qx", "qy", "qz"}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise ValueError("В файле ориентации нужны timestamp_us,qw,qx,qy,qz")
            for line_no, row in enumerate(reader, start=2):
                try:
                    records.append(AttitudeRecord(
                        int(row["timestamp_us"]),
                        float(row["qw"]), float(row["qx"]),
                        float(row["qy"]), float(row["qz"]),
                    ))
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Недопустимая ориентация в строке {line_no}") from exc
        return cls(records, **kwargs)

    def at_camera_time(self, camera_timestamp_us: int) -> tuple[float, float, float]:
        t = camera_timestamp_us + self._offset
        if not isinstance(camera_timestamp_us, int) or camera_timestamp_us <= 0:
            raise ValueError("Недопустимая временная метка кадра")
        idx = bisect.bisect_left(self._times, t)
        if idx < len(self._times) and self._times[idx] == t:
            return quaternion_to_rpy(self._quats[idx])
        if idx == 0 or idx == len(self._times):
            raise ValueError("Нет отсчетов ориентации по обе стороны кадра")
        t0, t1 = self._times[idx - 1], self._times[idx]
        if t1 - t0 > self._max_gap:
            raise ValueError("Разрыв журнала ориентации превышает допуск")
        q = slerp(self._quats[idx - 1], self._quats[idx],
                  (t - t0) / (t1 - t0))
        return quaternion_to_rpy(q)
