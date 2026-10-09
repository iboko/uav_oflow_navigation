"""Numerical validation of statistical consistency and gauge constraints."""
from dataclasses import replace
from math import cos, sin

import numpy as np
import pytest

from src.error_state_inertial_filter import EskfSnapshot
from src.eskf_consistency import (
    CHI2_2_95, LinearizationSample, TruthSample,
    flight_level_summary, nees_full_15, nees_horizontal_velocity,
    observability_velocity_only, quaternion_angular_error_rad,
    right_attitude_error_rotation_vector,
)


def state(t=1_000_000, vn=1.0, ve=0.0, variance=0.04):
    p = np.diag([2.] * 3 + [variance] * 3 +
                [0.01] * 3 + [0.0004] * 3 + [0.0025] * 3)
    return EskfSnapshot(
        timestamp_us=t, status="VISUAL_CORRECTED",
        position_ned_m=(0., 0., 0.),
        velocity_ned_m_s=(vn, ve, 0.),
        quaternion_body_to_ned=(1., 0., 0., 0.),
        gyro_bias_rad_s=(0., 0., 0.),
        accel_bias_m_s2=(0., 0., 0.),
        covariance_15x15=tuple(map(tuple, p)),
        last_nis=0.0, accepted_visual_count=1,
        rejected_visual_count=0
    )


def truth(t=1_000_000, vn=0.8, ve=0.0):
    return TruthSample(t, (vn, ve, 0.), "INDEPENDENT_MOTION_CAPTURE")


def test_velocity_nees_matches_analytical_mahalanobis_and_unit_contract():
    result = nees_horizontal_velocity(state(), truth())
    assert result["nees_vne_2d"] == pytest.approx(1.0, abs=1e-12)
    assert result["horizontal_speed_error_m_s"] == pytest.approx(0.2)
    assert result["inside_95pct_chi2_2"]


def test_nees_uses_correlated_2x2_marginal_and_is_not_diagonal_only():
    snap = state(vn=1., ve=1.)
    covariance = np.asarray(snap.covariance_15x15)
    covariance[3, 4] = covariance[4, 3] = 0.015
    snap = replace(snap, covariance_15x15=tuple(map(tuple, covariance)))
    outcome = nees_horizontal_velocity(snap, TruthSample(
        snap.timestamp_us, (0., 0., 0.), "external-camera-tracker"
    ))
    residual = np.array([1., 1.])
    expected = residual @ np.linalg.solve(covariance[3:5, 3:5], residual)
    assert outcome["nees_vne_2d"] == pytest.approx(expected)


def test_rejects_unsynchronised_or_non_independent_truth():
    with pytest.raises(ValueError, match="совпадают"):
        nees_horizontal_velocity(state(), truth(t=1_000_001))
    with pytest.raises(ValueError, match="независимый"):
        nees_horizontal_velocity(state(), TruthSample(
            1_000_000, (0., 0., 0.), "estimator"
        ))


def test_bad_or_singular_covariance_not_used_for_false_nees():
    snap = state()
    cov = np.asarray(snap.covariance_15x15)
    cov[3, 3] = -1
    with pytest.raises(ValueError, match="положительно определена"):
        nees_horizontal_velocity(replace(snap, covariance_15x15=tuple(map(tuple, cov))),
                                 truth())


def test_quaternion_error_is_sign_invariant_and_rotvec_right_error():
    q = (cos(0.1), 0., 0., sin(0.1))  # 0.2 rad yaw
    angle = quaternion_angular_error_rad((1., 0., 0., 0.), q)
    assert angle == pytest.approx(0.2, abs=1e-12)
    assert quaternion_angular_error_rad(
        (1., 0., 0., 0.), tuple(-v for v in q)
    ) == pytest.approx(0.2, abs=1e-12)
    rotvec = right_attitude_error_rotation_vector((1.,0.,0.,0.), q)
    assert np.allclose(rotvec, [0., 0., 0.2], atol=1e-12)


def test_full_15d_nees_requires_independent_biases_and_gauge_alignment():
    snap = state()
    full_truth = replace(truth(), position_ned_m=(0.,0.,0.),
                         quaternion_body_to_ned=(1.,0.,0.,0.))
    with pytest.raises(ValueError, match="подтвержденной"):
        nees_full_15(snap, full_truth,
            truth_gyro_bias_rad_s=(0.,0.,0.),
            truth_accel_bias_m_s2=(0.,0.,0.),
            origin_alignment_verified=False, heading_alignment_verified=True)
    value = nees_full_15(
        snap, full_truth,
        truth_gyro_bias_rad_s=(0.,0.,0.),
        truth_accel_bias_m_s2=(0.,0.,0.),
        origin_alignment_verified=True, heading_alignment_verified=True,
    )
    assert value == pytest.approx(1.0, abs=1e-11)


def test_velocity_only_measurements_cannot_identify_absolute_position():
    parked = [
        LinearizationSample(0.01, tuple(map(tuple, np.eye(3))),
                            (0.,0.,0.), (0.,0.,-9.80665))
        for _ in range(25)
    ]
    result = observability_velocity_only(parked)
    assert result["state_dimension"] == 15
    assert result["rank"] < 15
    assert result["unobservable_dimension"] >= 3
    assert result["translation_gauge_unobservable"]


def test_rotation_matrix_and_timestep_validations():
    invalid = LinearizationSample(0.07, tuple(map(tuple, np.eye(3))),
                                  (0.,0.,0.), (0.,0.,0.))
    with pytest.raises(ValueError, match="временной шаг"):
        observability_velocity_only([invalid])
    reflection = LinearizationSample(
        0.01, ((-1.,0.,0.),(0.,1.,0.),(0.,0.,1.)),
        (0.,0.,0.), (0.,0.,0.)
    )
    with pytest.raises(ValueError, match="поворотом"):
        observability_velocity_only([reflection])


def test_flight_bootstrap_uses_flight_as_sampling_unit_not_video_frames():
    data = {}
    for flight, e in (("F01", 0.1), ("F02", 0.3), ("F03", 0.5)):
        data[flight] = [
            nees_horizontal_velocity(state(1_000_000+i*10_000, vn=e),
                                     TruthSample(1_000_000+i*10_000,
                                                 (0.,0.,0.), "independent"))
            for i in range(35)
        ]
    summary = flight_level_summary(data)
    assert summary["independent_flights"] == 3
    assert summary["sufficient_for_preliminary_screening"]
    assert not summary["validated"]
    assert summary["flight_bootstrap_95pct_interval_for_mean"] is not None
    assert len(summary["flight_bootstrap_95pct_interval_for_mean"]) == 2
    assert summary["total_matched_samples"] == 105


def test_synthetic_one_flight_can_be_reported_but_not_validated():
    data = {"single": [nees_horizontal_velocity(state(), truth())]}
    summary = flight_level_summary(data)
    assert not summary["sufficient_for_preliminary_screening"]
    assert not summary["validated"]
    assert summary["flight_bootstrap_95pct_interval_for_mean"] is None
