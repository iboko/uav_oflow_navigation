"""Проверка отложенных визуальных коррекций с повторным прогнозом ИИМ."""
from math import sin

import numpy as np
import pytest

from src.error_state_inertial_filter import InertialErrorStateFilter, ImuNoiseDensity
from src.fixed_lag_eskf_replay import FixedLagEskfReplay
from src.horizontal_visual_inertial_filter import VisualVelocityMeasurement
from src.imu_preintegration import ImuReading, ImuBias


def factory():
    return InertialErrorStateFilter(
        quaternion_body_to_ned=(1., 0., 0., 0.),
        initial_bias=ImuBias(),
        noise=ImuNoiseDensity(0.003, 0.03, 0.00005, 0.0003),
        max_imu_dt_s=0.04,
    )


def imu(t, index):
    return ImuReading(t, (0., 0., 0.),
                      (0.4 + 0.1 * sin(index / 12.), 0., -9.80665))


def camera(t, vn, variance=0.01):
    return VisualVelocityMeasurement(
        t, vn, 0., ((variance, 0.), (0., variance))
    )


def simulate(arrival_lag_us, *, between_imu=False):
    wrapper = FixedLagEskfReplay(factory, max_lag_us=350000)
    latest = 0
    outcomes = []
    planned = []
    timestamps = [1_000_000 + i * 10_000 for i in range(151)]
    for i, t in enumerate(timestamps):
        if i > 0 and i % 10 == 0:
            exposure = t - (5000 if between_imu else 0)
            # Finite validation speed, not an external state reference.
            expected = 0.4 * (exposure - timestamps[0]) * 1e-6
            planned.append((exposure + arrival_lag_us, camera(exposure, expected)))
        wrapper.push_imu(imu(t, i))
        latest = t
        ready = [evt for evt in planned if evt[0] <= latest]
        for arrival, measurement in sorted(ready, key=lambda pair: pair[0]):
            result = wrapper.push_delayed_visual(measurement)
            outcomes.append(result.status)
            planned.remove((arrival, measurement))
    for arrival, measurement in planned:
        outcomes.append(wrapper.push_delayed_visual(measurement).status)
    return wrapper, outcomes


@pytest.mark.parametrize("between_imu", [False, True])
def test_delayed_replay_matches_in_order_measurements(between_imu):
    immediate, immediate_statuses = simulate(
        5000 if between_imu else 0, between_imu=between_imu
    )
    delayed, late_statuses = simulate(140000, between_imu=between_imu)
    assert immediate_statuses == late_statuses
    assert immediate.state.accepted_visual_count > 10
    a, b = immediate.state, delayed.state
    assert np.allclose(a.position_ned_m, b.position_ned_m, atol=1e-10)
    assert np.allclose(a.velocity_ned_m_s, b.velocity_ned_m_s, atol=1e-10)
    assert np.allclose(a.quaternion_body_to_ned, b.quaternion_body_to_ned, atol=1e-10)
    assert np.allclose(a.covariance_15x15, b.covariance_15x15, atol=1e-10)


def test_out_of_order_camera_arrival_rewinds_and_replays_in_measurement_time_order():
    baseline = FixedLagEskfReplay(factory)
    reordered = FixedLagEskfReplay(factory)
    frames = [1_000_000 + k * 10000 for k in range(31)]
    for i, t in enumerate(frames):
        baseline.push_imu(imu(t, i))
        reordered.push_imu(imu(t, i))
    v1 = camera(1_150_000, 0.06)
    v2 = camera(1_200_000, 0.08)
    assert baseline.push_delayed_visual(v1).status == "VISUAL_CORRECTED"
    assert baseline.push_delayed_visual(v2).status == "VISUAL_CORRECTED"
    assert reordered.push_delayed_visual(v2).status == "VISUAL_CORRECTED"
    assert reordered.push_delayed_visual(v1).status == "VISUAL_CORRECTED"
    assert np.allclose(baseline.state.position_ned_m,
                       reordered.state.position_ned_m, atol=1e-12)
    assert np.allclose(baseline.state.covariance_15x15,
                       reordered.state.covariance_15x15, atol=1e-12)


def test_late_measurement_is_rejected_and_preserves_state():
    engine = FixedLagEskfReplay(factory, max_lag_us=100_000)
    for i in range(41):
        engine.push_imu(imu(1_000_000 + i * 10000, i))
    before = engine.state
    result = engine.push_delayed_visual(camera(1_100_000, 0.))
    assert result.status == "VISUAL_DELAY_EXCEEDED"
    assert engine.state == before
    assert engine.stored_visual_count == 0


def test_duplicate_camera_sample_and_invalid_covariance_never_modify_state():
    engine = FixedLagEskfReplay(factory)
    for i in range(11):
        engine.push_imu(imu(1_000_000 + i * 10000, i))
    measurement = camera(1_060_000, 0.02)
    assert engine.push_delayed_visual(measurement).status == "VISUAL_CORRECTED"
    original = engine.state
    assert engine.push_delayed_visual(measurement).status == "REPEATED_VISUAL_MEASUREMENT"
    bad = camera(1_070_000, 0.03, variance=float("nan"))
    assert engine.push_delayed_visual(bad).status == "INVALID_VISUAL_MEASUREMENT"
    assert engine.state == original


def test_history_is_bounded_and_checkpoint_contains_prior_corrections():
    engine = FixedLagEskfReplay(factory, max_lag_us=50_000,
                                max_imu_samples=12)
    for i in range(101):
        timestamp = 1_000_000 + i * 10000
        engine.push_imu(imu(timestamp, i))
        if i and i % 3 == 0:
            m = camera(timestamp, 0.4 * (timestamp - 1_000_000) * 1e-6)
            assert engine.push_delayed_visual(m).status in (
                "VISUAL_CORRECTED", "VISUAL_OUTLIER_REJECTED"
            )
        assert engine.stored_imu_count <= 6
    assert engine.stored_visual_count <= 3
    assert engine.state.timestamp_us == 2_000_000
    assert engine.state.accepted_visual_count > 10


def test_multiple_camera_measurements_inside_single_imu_interval():
    engine = FixedLagEskfReplay(factory)
    engine.push_imu(imu(1_000_000, 0))
    engine.push_imu(imu(1_010_000, 1))
    last = camera(1_008_000, 0.)
    first = camera(1_003_000, 0.)
    engine.push_delayed_visual(last)
    engine.push_delayed_visual(first)
    assert engine.state.accepted_visual_count == 2
    assert np.linalg.eigvalsh(engine.state.covariance_15x15)[0] > 0.


def test_future_measurement_not_applied_prematurely_and_gap_does_not_rearm():
    engine = FixedLagEskfReplay(factory)
    engine.push_imu(imu(1_000_000, 0))
    assert engine.push_delayed_visual(camera(1_010_000, 0.)).status == (
        "FUTURE_VISUAL_MEASUREMENT"
    )
    with pytest.raises(ValueError, match="разрыв"):
        engine.push_imu(imu(1_100_000, 10))
    with pytest.raises(ValueError, match="Инерциальный фильтр"):
        engine.push_imu(imu(1_110_000, 11))


def test_nis_kept_for_each_exposure_even_after_later_replay():
    engine = FixedLagEskfReplay(factory, max_lag_us=300_000)
    for i in range(31):
        engine.push_imu(imu(1_000_000 + i * 10000, i))
    outlier = engine.push_delayed_visual(camera(1_210_000, 100., variance=0.001))
    assert outlier.status == "VISUAL_OUTLIER_REJECTED"
    assert np.isfinite(outlier.nis)
    assert outlier.nis > 9.210340371976184

    earlier = engine.push_delayed_visual(camera(1_160_000, 0.07))
    assert earlier.status == "VISUAL_CORRECTED"
    assert np.isfinite(earlier.nis)
    assert 0 <= earlier.nis < 9.210340371976184
    # The outlier remains an excluded measurement after the rewind.
    assert engine.state.rejected_visual_count == 1
