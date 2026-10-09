"""Проверка численной схемы интегрирования ИИМ и смещений нуля."""
from math import cos, sin
import numpy as np
import pytest

from src.imu_preintegration import (
    GRAVITY_NED, ImuBias, ImuPreintegrator, ImuReading,
    calibrate_stationary_bias, rotation_body_to_ned, quat_exponential
)


def reading(t, gyro=(0., 0., 0.), accel=(0., 0., -9.80665)):
    return ImuReading(t, gyro, accel)


def test_quaternion_exponential_is_normalized_and_rotation_valid():
    q = quat_exponential(np.array([0.02, -0.01, 0.1]))
    assert np.linalg.norm(q) == pytest.approx(1.0, abs=1e-13)
    r = rotation_body_to_ned(q)
    assert np.allclose(r.T @ r, np.eye(3), atol=1e-12)
    assert np.linalg.det(r) == pytest.approx(1.0, abs=1e-12)


def test_stationary_imu_does_not_generate_vertical_or_horizontal_drift():
    imu = ImuPreintegrator(quaternion_body_to_ned=(1., 0., 0., 0.))
    for k in range(101):
        imu.step(reading(1_000_000 + k * 10_000))
    assert np.linalg.norm(imu.vel) < 1e-12
    assert np.linalg.norm(imu.pos) < 1e-12


def test_constant_horizontal_acceleration_integrates_position_and_speed():
    imu = ImuPreintegrator(quaternion_body_to_ned=(1., 0., 0., 0.))
    for k in range(101):
        imu.step(reading(1_000_000 + k * 10_000,
                         accel=(2., 0., -9.80665)))
    assert imu.vel[0] == pytest.approx(2., abs=1e-10)
    assert imu.pos[0] == pytest.approx(1., abs=1e-10)


def test_yaw_rate_integrates_to_known_angle_with_static_bias():
    rate = 0.4
    bias = ImuBias(gyro_body_rad_s=(0., 0., 0.1))
    imu = ImuPreintegrator(quaternion_body_to_ned=(1., 0., 0., 0.), bias=bias)
    for k in range(101):
        imu.step(reading(1_000_000 + k * 10_000,
                         gyro=(0., 0., rate + 0.1)))
    assert imu.q[0] == pytest.approx(cos(rate / 2), abs=1e-10)
    assert imu.q[3] == pytest.approx(sin(rate / 2), abs=1e-10)


def test_stationary_bias_calibration_needs_explicit_standstill_and_orientation():
    bias = ImuBias(
        gyro_body_rad_s=(0.003, -0.002, 0.004),
        accel_body_m_s2=(0.08, -0.04, 0.06),
    )
    samples = [
        reading(1_000_000 + i * 10_000, bias.gyro_body_rad_s,
                (bias.accel_body_m_s2[0], bias.accel_body_m_s2[1],
                 -9.80665 + bias.accel_body_m_s2[2]))
        for i in range(50)
    ]
    with pytest.raises(ValueError, match="подтверждена"):
        calibrate_stationary_bias(
            samples, known_attitude_body_to_ned=(1., 0., 0., 0.),
            confirmed_stationary=False
        )
    estimated = calibrate_stationary_bias(
        samples, known_attitude_body_to_ned=(1., 0., 0., 0.),
        confirmed_stationary=True
    )
    assert np.allclose(estimated.gyro_body_rad_s, bias.gyro_body_rad_s, atol=1e-12)
    assert np.allclose(estimated.accel_body_m_s2, bias.accel_body_m_s2, atol=1e-12)
    imu = ImuPreintegrator(
        quaternion_body_to_ned=(1., 0., 0., 0.), bias=estimated
    )
    for s in samples:
        imu.step(s)
    assert np.linalg.norm(imu.vel) < 1e-10


def test_inertial_gap_latches_failure_until_reset():
    imu = ImuPreintegrator(quaternion_body_to_ned=(1., 0., 0., 0.))
    imu.step(reading(1_000_000))
    with pytest.raises(ValueError, match="непрерывности"):
        imu.step(reading(1_100_000))
    assert not imu.healthy
    with pytest.raises(ValueError, match="reset"):
        imu.step(reading(1_110_000))
    imu.reset(quaternion_body_to_ned=(1., 0., 0., 0.))
    assert imu.healthy
    assert imu.step(reading(1_200_000)).dt_s == 0.


def test_nonfinite_measurements_and_invalid_attitude_fail():
    with pytest.raises(ValueError):
        ImuPreintegrator(quaternion_body_to_ned=(0., 0., 0., 0.))
    imu = ImuPreintegrator(quaternion_body_to_ned=(1., 0., 0., 0.))
    with pytest.raises(ValueError, match="Неконечный"):
        imu.step(reading(1_000_000, gyro=(0., 0., float("nan"))))


def test_pitched_stationary_imu_cancels_gravity_in_ned():
    pitch = 0.2
    # q body FRD -> NED, rotate about +Y.
    q = (cos(pitch / 2), 0., sin(pitch / 2), 0.)
    f_body = rotation_body_to_ned(q).T @ -GRAVITY_NED
    imu = ImuPreintegrator(quaternion_body_to_ned=q)
    for i in range(101):
        imu.step(reading(1_000_000 + i * 10_000,
                         accel=tuple(f_body)))
    assert np.linalg.norm(imu.vel) < 1e-9


def test_nonfinite_imu_is_latched_until_explicit_reset():
    imu = ImuPreintegrator(quaternion_body_to_ned=(1., 0., 0., 0.))
    imu.step(reading(1_000_000))
    with pytest.raises(ValueError, match="Неконечный"):
        imu.step(reading(1_010_000, accel=(float("nan"), 0., -9.80665)))
    assert not imu.healthy
    with pytest.raises(ValueError, match="reset"):
        imu.step(reading(1_020_000))
    imu.reset(quaternion_body_to_ned=(1., 0., 0., 0.))
    assert imu.step(reading(1_030_000)).dt_s == 0.0
