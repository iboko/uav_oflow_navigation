"""Строгие численные регрессии 15-мерного РФК ошибок состояния."""
from math import cos, sin

import numpy as np
import pytest

from src.error_state_inertial_filter import (
    ImuNoiseDensity, InertialErrorStateFilter,
    continuous_error_jacobian, discrete_process_covariance,
    exp_transition, skew,
)
from src.horizontal_visual_inertial_filter import VisualVelocityMeasurement
from src.imu_preintegration import (
    GRAVITY_NED, ImuBias, ImuReading, quat_exponential,
    rotation_body_to_ned,
)


def noise():
    return ImuNoiseDensity(
        gyro_rad_s_sqrt_hz=0.003,
        accel_m_s2_sqrt_hz=0.03,
        gyro_bias_walk_rad_s2_sqrt_hz=0.00005,
        accel_bias_walk_m_s3_sqrt_hz=0.0003,
    )


def filt(q=(1., 0., 0., 0.), bias=ImuBias()):
    return InertialErrorStateFilter(
        quaternion_body_to_ned=q,
        initial_bias=bias, noise=noise()
    )


def sample(time_us, ax=0., az=-9.80665, gyro=(0., 0., 0.)):
    return ImuReading(time_us, gyro, (ax, 0., az))


def visual(time_us, vn=0., ve=0., var=0.0025):
    return VisualVelocityMeasurement(
        time_us, vn, ve, ((var, 0.), (0., var))
    )


def test_stationary_imu_cancels_gravity_and_covariance_grows():
    ekf = filt()
    original = np.asarray(ekf.snapshot().covariance_15x15)
    for k in range(101):
        state = ekf.predict(sample(1_000_000 + 10_000 * k))
    assert state.status == "INERTIAL_ONLY"
    assert np.linalg.norm(state.velocity_ned_m_s) < 1e-10
    assert np.linalg.norm(state.position_ned_m) < 1e-10
    cov = np.asarray(state.covariance_15x15)
    assert cov.shape == (15, 15)
    assert np.allclose(cov, cov.T, atol=1e-9)
    assert np.linalg.eigvalsh(cov)[0] > 0.
    assert cov[3, 3] > original[3, 3]


def test_constant_horizontal_acceleration_midpoint_motion():
    ekf = filt()
    for k in range(101):
        out = ekf.predict(sample(1_000_000 + 10_000 * k, ax=2.))
    assert out.velocity_ned_m_s[0] == pytest.approx(2., abs=1e-9)
    assert out.position_ned_m[0] == pytest.approx(1., abs=1e-9)


def test_pitched_stationary_frame_uses_correct_specific_force():
    pitch = 0.2
    q = (cos(pitch / 2), 0., sin(pitch / 2), 0.)
    f = rotation_body_to_ned(q).T @ -GRAVITY_NED
    ekf = filt(q=q)
    for k in range(101):
        out = ekf.predict(ImuReading(
            1_000_000 + 10_000 * k, (0., 0., 0.), tuple(f)
        ))
    assert np.linalg.norm(out.velocity_ned_m_s) < 1e-8


def test_analytic_attitude_jacobian_matches_finite_difference():
    q = quat_exponential(np.array([0.3, -0.2, 0.1]))
    r = rotation_body_to_ned(q)
    force = np.array([2., -1., -9.8])
    a = continuous_error_jacobian(r, np.array([0.1, -0.05, 0.02]), force)
    eps = 1e-7
    for j in range(3):
        direction = np.eye(3)[j] * eps
        perturbed = rotation_body_to_ned(
            np.array([
                q[0]*quat_exponential(direction)[0] -
                np.dot(q[1:], quat_exponential(direction)[1:]),
                *(q[0]*quat_exponential(direction)[1:] +
                  quat_exponential(direction)[0]*q[1:] +
                  np.cross(q[1:], quat_exponential(direction)[1:]))
            ])
        )
        actual = (perturbed @ force - r @ force) / eps
        assert np.allclose(actual, a[3:6, 6+j], atol=1e-6)
    assert np.allclose(a[3:6, 12:15], -r)
    assert np.allclose(a[6:9, 9:12], -np.eye(3))


def test_process_noise_has_expected_position_velocity_cross_terms():
    r = np.eye(3)
    f = continuous_error_jacobian(r, np.zeros(3), np.zeros(3))
    dt = 0.01
    qd = discrete_process_covariance(f, r, dt, noise())
    variance_density = noise().accel_m_s2_sqrt_hz**2
    assert qd[3, 3] == pytest.approx(variance_density * dt, rel=1e-8)
    assert qd[0, 3] == pytest.approx(variance_density * dt**2 / 2, rel=1e-8)
    assert qd[0, 0] == pytest.approx(variance_density * dt**3 / 3, rel=1e-8)
    assert np.linalg.eigvalsh(qd)[0] >= -1e-14


def test_visual_corrects_velocity_using_joseph_and_preserves_spd():
    ekf = filt()
    for k in range(101):
        t = 1_000_000 + 10_000*k
        ekf.predict(sample(t))
        if k and k % 10 == 0:
            state = ekf.update_visual_velocity(visual(t))
            assert state.status == "VISUAL_CORRECTED"
    state = ekf.snapshot()
    assert state.accepted_visual_count == 10
    assert np.linalg.norm(state.velocity_ned_m_s) < 1e-9
    assert np.linalg.eigvalsh(np.array(state.covariance_15x15))[0] > 0.


def test_measurement_outlier_is_rejected_without_state_jump():
    ekf = filt()
    ekf.predict(sample(1_000_000))
    ekf.predict(sample(1_010_000))
    previous = ekf.snapshot()
    out = ekf.update_visual_velocity(visual(1_010_000, vn=100.))
    assert out.status == "VISUAL_OUTLIER_REJECTED"
    assert out.rejected_visual_count == 1
    assert out.velocity_ned_m_s == previous.velocity_ned_m_s
    assert out.last_nis > 9.21


def test_bad_time_is_not_silently_repropagated():
    ekf = filt()
    ekf.predict(sample(1_000_000))
    ekf.predict(sample(1_010_000))
    out = ekf.update_visual_velocity(visual(1_000_000))
    assert out.status == "UNALIGNED_VISUAL_TIMESTAMP"
    assert out.accepted_visual_count == 0
    # Fresh IMU is still allowed; stale visual measure did not corrupt it.
    assert ekf.predict(sample(1_020_000)).status == "INERTIAL_ONLY"


def test_nonfinite_imu_and_gap_fail_closed():
    ekf = filt()
    ekf.predict(sample(1_000_000))
    failed = ekf.predict(sample(1_100_000))
    assert failed.status == "IMU_INVALID_OR_GAP"
    assert ekf.predict(sample(1_110_000)).status == "IMU_INVALID_OR_GAP"
    ekf2 = filt()
    assert ekf2.predict(sample(1_000_000, ax=float("nan"))).status == "IMU_INVALID_OR_GAP"


def test_invalid_covariance_must_not_fuse():
    ekf = filt()
    ekf.predict(sample(1_000_000))
    bad = VisualVelocityMeasurement(
        1_000_000, 0., 0., ((0.1, 0.3), (0.3, 0.1))
    )
    out = ekf.update_visual_velocity(bad)
    assert out.status == "VISUAL_COVARIANCE_INVALID"
    assert out.accepted_visual_count == 0


def test_gyro_bias_is_explicit_state_not_claimed_observable_at_hover():
    bias = ImuBias(gyro_body_rad_s=(0., 0., 0.08))
    ekf = filt(bias=bias)
    for k in range(101):
        out = ekf.predict(sample(
            1_000_000 + 10_000 * k, gyro=(0., 0., 0.08)
        ))
    assert out.quaternion_body_to_ned[0] == pytest.approx(1., abs=1e-10)
    assert out.gyro_bias_rad_s[2] == pytest.approx(0.08)
    assert out.accepted_visual_count == 0


def test_horizontal_velocity_cannot_make_all_states_observable_at_hover():
    # Stationary FRD aligned with NED: a single 2D velocity measurement
    # has no direct sensitivity to yaw or z gyro bias in this trajectory.
    a = continuous_error_jacobian(
        np.eye(3), np.zeros(3), np.array([0., 0., -9.80665])
    )
    h = np.zeros((2, 15))
    h[:, 3:5] = np.eye(2)
    observability = []
    power = np.eye(15)
    for _ in range(15):
        observability.append(h @ power)
        power = power @ a
    o = np.vstack(observability)
    assert np.linalg.matrix_rank(o) < 15
    assert np.allclose(o[:, 8], 0.)    # yaw perturbation
    assert np.allclose(o[:, 11], 0.)   # z gyro-bias perturbation


def test_small_visual_velocity_update_can_correct_attitude_coupling():
    pitch = 0.05
    # Incorrect nominal pitch for a truly stationary, level platform.
    ekf = filt(q=(cos(pitch / 2), 0., sin(pitch / 2), 0.))
    for k in range(21):
        t = 1_000_000 + k * 10_000
        before = ekf.predict(sample(t))
    pre = before.velocity_ned_m_s[0]
    assert abs(pre) > 0.01
    updated = ekf.update_visual_velocity(visual(t, var=0.0001))
    assert updated.status == "VISUAL_CORRECTED"
    assert abs(updated.velocity_ned_m_s[0]) < abs(pre)
    assert np.linalg.norm(updated.quaternion_body_to_ned) == pytest.approx(
        1.0, abs=1e-12
    )
    assert np.linalg.eigvalsh(np.asarray(updated.covariance_15x15))[0] > 0.
