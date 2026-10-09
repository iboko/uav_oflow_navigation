"""РФК по сырым ИИМ и визуальной скорости: математические регрессии."""
import numpy as np
import pytest

from src.horizontal_visual_inertial_filter import (
    HorizontalVisualInertialFilter, VisualVelocityMeasurement
)
from src.imu_preintegration import ImuBias, ImuReading


def filt():
    return HorizontalVisualInertialFilter(
        quaternion_body_to_ned=(1., 0., 0., 0.),
        initial_bias=ImuBias(),
        gyro_noise_rad_s=0.004,
        accel_noise_m_s2=0.08,
        accel_bias_walk_m_s2_sqrt_s=0.002,
    )


def reading(t, ax=0., az=-9.80665):
    return ImuReading(t, (0., 0., 0.), (ax, 0., az))


def vision(t, v_n=0., v_e=0., variance=0.0025):
    return VisualVelocityMeasurement(
        t, v_n, v_e, ((variance, 0.), (0., variance))
    )


def test_stationary_imu_and_visual_give_no_false_motion():
    ekf = filt()
    for k in range(101):
        timestamp = 1_000_000 + 10_000 * k
        ekf.predict(reading(timestamp))
        if k and k % 10 == 0:
            state = ekf.update_visual(vision(timestamp))
            assert state.status == "VISUAL_CORRECTED", state
    result = ekf.snapshot()
    assert result.visual_updates == 10
    assert abs(result.north_m) < 1e-12
    assert abs(result.east_m) < 1e-12
    assert abs(result.vn_m_s) < 1e-12
    assert np.linalg.eigvalsh(result.covariance_6x6).min() > 0.


def test_constant_acceleration_predicts_displacement_and_velocity():
    ekf = filt()
    for k in range(101):
        ekf.predict(reading(1_000_000 + 10_000 * k, ax=1.))
    result = ekf.snapshot()
    assert result.vn_m_s == pytest.approx(1., abs=1e-10)
    assert result.north_m == pytest.approx(0.5, abs=1e-10)


def test_visual_innovation_gate_blocks_extreme_false_speed():
    ekf = filt()
    ekf.predict(reading(1_000_000))
    ekf.predict(reading(1_010_000))
    before = ekf.snapshot()
    rejected = ekf.update_visual(vision(1_010_000, v_n=300.))
    assert rejected.status == "VISUAL_OUTLIER_REJECTED"
    assert rejected.visual_updates == 0
    assert rejected.vn_m_s == before.vn_m_s


def test_visual_measurement_timestamp_must_match_propagated_state():
    ekf = filt()
    ekf.predict(reading(1_000_000))
    ekf.predict(reading(1_010_000))
    rejected = ekf.update_visual(vision(1_000_000))
    assert rejected.status == "UNALIGNED_VISUAL_TIMESTAMP"
    assert rejected.visual_updates == 0
    corrected = ekf.update_visual(vision(1_010_000))
    assert corrected.status == "VISUAL_CORRECTED"
    duplicate = ekf.update_visual(vision(1_010_000))
    assert duplicate.status == "REPEATED_VISUAL_MEASUREMENT"
    assert duplicate.visual_updates == 1


def test_invalid_covariance_is_rejected_without_correction():
    ekf = filt()
    ekf.predict(reading(1_000_000))
    ekf.predict(reading(1_010_000))
    measurement = VisualVelocityMeasurement(
        1_010_000, 0., 0., ((0.2, 0.3), (0.3, 0.2))
    )
    assert ekf.update_visual(measurement).status == "VISUAL_COVARIANCE_INVALID"
    assert ekf.snapshot().visual_updates == 0


def test_bias_state_observed_with_repeated_external_velocity_corrections():
    ekf = filt()
    for k in range(1001):
        t = 1_000_000 + k * 10_000
        ekf.predict(reading(t, ax=0.08))
        if k and k % 10 == 0:
            ekf.update_visual(vision(t, variance=0.0025))
    result = ekf.snapshot()
    assert result.visual_updates >= 80
    assert result.accel_bias_x_m_s2 == pytest.approx(0.08, abs=0.03)
    assert abs(result.vn_m_s) < 0.08


def test_imu_time_gap_does_not_silently_restart_estimator():
    ekf = filt()
    ekf.predict(reading(1_000_000))
    state = ekf.predict(reading(1_100_000))
    assert state.status == "IMU_INVALID_OR_GAP"
    later = ekf.update_visual(vision(1_100_000))
    assert later.status == "IMU_INVALID_OR_GAP"
    assert later.visual_updates == 0


def test_filter_does_not_pretend_to_have_absolute_position():
    ekf = filt()
    result = ekf.snapshot()
    assert result.visual_updates == 0
    assert result.status == "WAITING_FOR_IMU"
    assert result.north_m == 0. and result.east_m == 0.
    # Coordinates are relative to the arbitrary initialization origin.
