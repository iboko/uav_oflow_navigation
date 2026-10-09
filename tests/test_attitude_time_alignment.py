"""Проверка синхронизации видеокадров с ориентацией ИНС."""
import csv
from math import cos, pi, sin

import numpy as np
import pytest

from src.attitude_time_alignment import (
    AttitudeRecord, AttitudeTimeline, quaternion_to_rpy, slerp
)


def yaw_record(t, yaw):
    return AttitudeRecord(t, cos(yaw / 2), 0., 0., sin(yaw / 2))


def test_slerp_interpolates_yaw_in_ned():
    timeline = AttitudeTimeline([
        yaw_record(1_000_000, 0.),
        yaw_record(1_020_000, pi / 2),
    ])
    roll, pitch, yaw = timeline.at_camera_time(1_010_000)
    assert roll == pytest.approx(0., abs=1e-12)
    assert pitch == pytest.approx(0., abs=1e-12)
    assert yaw == pytest.approx(pi / 4, abs=1e-10)


def test_slerp_short_path_across_yaw_wrap_and_quaternion_sign():
    timeline = AttitudeTimeline([
        yaw_record(1_000_000, pi - 0.1),
        yaw_record(1_020_000, -pi + 0.1),
    ])
    _, _, yaw = timeline.at_camera_time(1_010_000)
    assert abs(abs(yaw) - pi) < 1e-9
    q = np.array([0.3, 0.1, 0.1, 0.9])
    q /= np.linalg.norm(q)
    assert np.allclose(slerp(q, -q, 0.42), q)


def test_camera_clock_offset_is_applied_before_interpolation():
    timeline = AttitudeTimeline([
        yaw_record(1_000_000, 0.), yaw_record(1_020_000, pi / 2)
    ], camera_to_attitude_offset_us=10_000)
    _, _, yaw = timeline.at_camera_time(1_000_000)
    assert yaw == pytest.approx(pi / 4, abs=1e-10)


@pytest.mark.parametrize("t", [900_000, 1_030_000])
def test_no_extrapolation_outside_known_attitude(t):
    timeline = AttitudeTimeline([
        yaw_record(1_000_000, 0.), yaw_record(1_020_000, 0.2)
    ])
    with pytest.raises(ValueError, match="Нет отсчетов"):
        timeline.at_camera_time(t)


def test_missing_attitude_interval_is_not_silently_interpolated():
    timeline = AttitudeTimeline([
        yaw_record(1_000_000, 0.), yaw_record(1_150_000, 0.3)
    ], max_sample_gap_us=25_000)
    with pytest.raises(ValueError, match="Разрыв"):
        timeline.at_camera_time(1_075_000)


def test_duplicate_time_and_bad_quaternions_fail_closed():
    with pytest.raises(ValueError, match="возрастать"):
        AttitudeTimeline([
            yaw_record(1_000_000, 0.), yaw_record(1_000_000, 0.)
        ])
    with pytest.raises(ValueError, match="нормирован"):
        AttitudeTimeline([AttitudeRecord(1_000_000, 0., 0., 0., 0.)])
    with pytest.raises(ValueError, match="компоненты"):
        AttitudeTimeline([AttitudeRecord(1_000_000, float("nan"), 0., 0., 0.)])


def test_csv_rejects_missing_columns(tmp_path):
    p = tmp_path / "att.csv"
    with p.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp_us", "qw"])
        writer.writerow([1_000_000, 1])
    with pytest.raises(ValueError, match="нужны"):
        AttitudeTimeline.from_csv(str(p))
