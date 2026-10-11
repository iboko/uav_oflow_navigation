"""Коррекции ESKF по скорости в осях корпуса, курсу, карте и выставка."""
from math import atan2, cos, pi, sin

import numpy as np

from src.error_state_inertial_filter import (
    BodyVelocityMeasurement, ImuNoiseDensity, InertialErrorStateFilter,
    MapPositionMeasurement, YawMeasurement, choose_ci_weight, static_alignment,
)
from src.fixed_lag_eskf_replay import FixedLagEskfReplay
from src.horizontal_visual_inertial_filter import VisualVelocityMeasurement
from src.imu_preintegration import ImuBias, ImuReading

NOISE = ImuNoiseDensity(.003, .03, .00005, .0003)
STILL = (0., 0., -9.80665)


def make(q=(1., 0., 0., 0.), sigma=None, bias=ImuBias()):
    kwargs = {}
    if sigma is not None:
        kwargs["initial_error_sigma"] = sigma
    return InertialErrorStateFilter(quaternion_body_to_ned=q, initial_bias=bias,
                                    noise=NOISE, **kwargs)


def yaw_of(q):
    w, x, y, z = q
    return atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def default_sigma(**changes):
    sigma = [10., 10., 10., 1., 1., 1., .1, .1, .1, .02, .02, .02, .2, .2, .2]
    for index, value in changes.items():
        sigma[int(index[1:])] = value
    return tuple(sigma)


def test_body_velocity_makes_yaw_observable_without_external_attitude():
    f = make(sigma=default_sigma(s8=.5))
    true_yaw = .2
    for i in range(201):
        t = 1_000_000 + 10_000 * i
        f.predict(ImuReading(t, (0., 0., 0.), STILL))
        if i % 10 == 0:
            assert f.update_visual_velocity(VisualVelocityMeasurement(
                t, 1., 0., ((.0004, 0.), (0., .0004)))).status == "VISUAL_CORRECTED"
            assert f.update_body_velocity(BodyVelocityMeasurement(
                t, cos(true_yaw), -sin(true_yaw),
                ((.0004, 0.), (0., .0004)))).status == "VISUAL_CORRECTED"
    assert abs(yaw_of(f.q) - true_yaw) < .02
    assert f.covariance[8, 8] < .01


def test_body_velocity_jacobian_matches_finite_difference():
    f = make(q=(cos(.3), .05, -.04, sin(.3)))
    f.predict(ImuReading(1_000_000, (0., 0., 0.), STILL))
    f.v = np.array([1.2, -.4, .1])
    from src.error_state_inertial_filter import TH, V, skew
    from src.imu_preintegration import quat_exponential, quat_product, rotation_body_to_ned

    def h(v, q):
        return (rotation_body_to_ned(q).T @ v)[:2]

    rot = rotation_body_to_ned(f.q)
    analytic = np.zeros((2, 15))
    analytic[:, V] = rot.T[:2, :]
    analytic[:, TH] = skew(rot.T @ f.v)[:2, :]
    numeric = np.zeros((2, 15))
    eps = 1e-7
    for k in range(3):
        dv = np.zeros(3)
        dv[k] = eps
        numeric[:, 3 + k] = (h(f.v + dv, f.q) - h(f.v - dv, f.q)) / (2 * eps)
        q_plus = quat_product(f.q, quat_exponential(dv))
        q_minus = quat_product(f.q, quat_exponential(-dv))
        numeric[:, 6 + k] = (h(f.v, q_plus) - h(f.v, q_minus)) / (2 * eps)
    assert np.allclose(analytic, numeric, atol=1e-7)


def test_yaw_jacobian_matches_finite_difference_and_wraps():
    from src.imu_preintegration import quat_exponential, quat_product, rotation_body_to_ned
    q = np.array((cos(.1) * cos(.2), sin(.1) * cos(.2), -sin(.08), sin(.2) * cos(.1)))
    q = q / np.linalg.norm(q)

    def yaw(qq):
        r = rotation_body_to_ned(qq)
        return atan2(r[1, 0], r[0, 0])

    r = rotation_body_to_ned(q)
    den = r[0, 0] ** 2 + r[1, 0] ** 2
    analytic = np.array([0., (r[1, 0] * r[0, 2] - r[0, 0] * r[1, 2]) / den,
                         (r[0, 0] * r[1, 1] - r[1, 0] * r[0, 1]) / den])
    eps = 1e-7
    numeric = np.zeros(3)
    for k in range(3):
        d = np.zeros(3)
        d[k] = eps
        numeric[k] = (yaw(quat_product(q, quat_exponential(d))) -
                      yaw(quat_product(q, quat_exponential(-d)))) / (2 * eps)
    assert np.allclose(analytic, numeric, atol=1e-7)
    f = make(sigma=default_sigma(s8=1.))
    f.predict(ImuReading(1_000_000, (0., 0., 0.), STILL))
    out = f.update_yaw(YawMeasurement(1_000_000, 2 * pi - .05, 1e-6))
    assert out.status == "YAW_CORRECTED"
    assert abs(yaw_of(f.q) + .05) < 1e-3


def test_map_position_covariance_intersection_is_conservative():
    independent, ci = make(), make()
    for f in (independent, ci):
        f.predict(ImuReading(1_000_000, (0., 0., 0.), STILL))
        f.predict(ImuReading(1_010_000, (0., 0., 0.), STILL))
    prior = independent.covariance[0, 0]
    r = ((9., 0.), (0., 9.))
    assert independent.update_map_position(
        MapPositionMeasurement(1_010_000, 1., -1., r)).status == "POSITION_CORRECTED"
    assert ci.update_map_position(
        MapPositionMeasurement(1_010_000, 1., -1., r, 0.)).status == "POSITION_CORRECTED"
    assert 0. < ci.last_ci_weight < 1.
    assert independent.covariance[0, 0] < ci.covariance[0, 0] < prior
    far = make()
    far.predict(ImuReading(1_000_000, (0., 0., 0.), STILL))
    assert far.update_map_position(MapPositionMeasurement(
        1_000_000, 500., 0., ((1., 0.), (0., 1.)))).status == "POSITION_OUTLIER_REJECTED"
    # Grid optimum equals brute force det of the FULL 15x15 posterior.
    base = make()
    base.predict(ImuReading(1_000_000, (0., 0., 0.), STILL))
    p = base.covariance
    h = np.zeros((2, 15))
    h[:, 0:2] = np.eye(2)
    r2 = np.array([[1., -.3], [-.3, 5.]])
    def posterior_logdet(w):
        pw, rw = p / w, r2 / (1 - w)
        k = pw @ h.T @ np.linalg.inv(h @ pw @ h.T + rw)
        a = np.eye(15) - k @ h
        return np.linalg.slogdet(a @ pw @ a.T + k @ rw @ k.T)[1]
    brute = min(range(5, 96), key=lambda step: posterior_logdet(step / 100)) / 100
    assert choose_ci_weight(h @ p @ h.T, r2) == brute
    assert brute > .5  # full-state criterion does not inflate everything by 1/w


def test_static_alignment_recovers_tilt_and_gyro_bias():
    roll, pitch, g = .1, -.05, 9.80665
    bias = (.01, -.02, .005)
    readings = []
    for i in range(200):
        n = .01 * sin(1.7 * i)
        readings.append(ImuReading(
            1_000_000 + 5000 * i,
            (bias[0] + .001 * sin(2.1 * i), bias[1], bias[2]),
            (g * sin(pitch) + n, -g * sin(roll) * cos(pitch),
             -g * cos(roll) * cos(pitch) - n)))
    result = static_alignment(readings, yaw_rad=.7, yaw_sigma_rad=.05)
    assert result.status == "ALIGNED"
    w, x, y, z = result.quaternion_body_to_ned
    assert abs(atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)) - roll) < 1e-3
    assert abs(np.arcsin(2 * (w * y - z * x)) - pitch) < 1e-3
    assert abs(yaw_of(result.quaternion_body_to_ned) - .7) < 1e-9
    assert np.allclose(result.gyro_bias_rad_s, bias, atol=1e-4)
    assert result.initial_error_sigma[8] == .05
    f = make(q=result.quaternion_body_to_ned,
             sigma=result.initial_error_sigma,
             bias=ImuBias(result.gyro_bias_rad_s))
    for i, r in enumerate(readings):
        f.predict(ImuReading(2_000_000 + 5000 * i, r.gyro_body_rad_s,
                             r.specific_force_body_m_s2))
    assert np.max(np.abs(f.v)) < .02
    moved = list(readings)
    moved[50] = ImuReading(moved[50].timestamp_us, moved[50].gyro_body_rad_s,
                           (5., 0., -9.8))
    assert static_alignment(moved, yaw_rad=0., yaw_sigma_rad=.1).status == "NOT_STATIONARY"
    assert static_alignment(readings[:10], yaw_rad=0.,
                            yaw_sigma_rad=.1).status == "TOO_FEW_SAMPLES"


def test_frame_replay_equals_sequential_filter_for_all_kinds():
    sequential = make()
    replay = FixedLagEskfReplay(make, max_lag_us=600_000, max_imu_samples=200)
    frames = {}
    for k in range(4):
        t = 1_000_000 + 10_000 * (20 + 12 * k)
        frames[t] = [
            VisualVelocityMeasurement(t, .01 * k, -.02, ((.04, 0.), (0., .04))),
            BodyVelocityMeasurement(t, .012 * k, -.018, ((.05, 0.), (0., .05))),
            YawMeasurement(t, .001 * k, .01),
            MapPositionMeasurement(t, .03 * k, -.01 * k, ((4., 0.), (0., 4.)), 0.),
        ]
    for i in range(90):
        t = 1_000_000 + 10_000 * i
        s = .01 * i
        imu = ImuReading(t, (.004 * sin(.9 * s), -.003 * cos(.7 * s), .03 * cos(.5 * s)),
                         (.3 * sin(.6 * s), .2 * cos(.4 * s), -9.80665 + .05 * sin(.8 * s)))
        sequential.predict(imu)
        for m in frames.get(t, []):
            sequential.apply(m)
        replay.push_imu(imu)
        if i == 75:
            for stamp in sorted(frames, reverse=True):
                results = replay.push_delayed_frame(frames[stamp])
                assert all(r.status.endswith(("CORRECTED", "REJECTED")) for r in results)
    state = replay.state
    assert np.allclose(state.position_ned_m, sequential.p, atol=1e-9)
    assert np.allclose(state.velocity_ned_m_s, sequential.v, atol=1e-9)
    assert np.allclose(state.covariance_15x15, sequential.covariance, atol=1e-9)


def test_frame_duplicate_kind_and_ci_on_velocity_are_rejected():
    replay = FixedLagEskfReplay(make, max_lag_us=400_000)
    for i in range(31):
        replay.push_imu(ImuReading(1_000_000 + 10_000 * i, (0., 0., 0.), STILL))
    v = VisualVelocityMeasurement(1_200_000, 0., 0., ((.04, 0.), (0., .04)))
    first, second = replay.push_delayed_frame([v, v])
    assert first.status == "VISUAL_CORRECTED"
    assert second.status == "INVALID_VISUAL_MEASUREMENT"
    bad = MapPositionMeasurement(1_210_000, 0., 0., ((4., 0.), (0., 4.)), 1.5)
    assert replay.push_delayed_visual(bad).status == "INVALID_VISUAL_MEASUREMENT"
    assert replay.stored_visual_count == 1


def test_ground_point_jacobian_matches_finite_difference():
    from src.error_state_inertial_filter import MapGroundPointMeasurement, TH, skew
    from src.imu_preintegration import quat_exponential, quat_product, rotation_body_to_ned
    q = np.array((cos(.05) * cos(.3), sin(.05), sin(.04), sin(.3)))
    q = q / np.linalg.norm(q)
    d = np.array((.1, -.05, 1.))
    d = d / np.linalg.norm(d)
    height = 100.

    def offset(qq):
        dn = rotation_body_to_ned(qq) @ d
        return height * dn[:2] / dn[2]

    rot = rotation_body_to_ned(q)
    dn = rot @ d
    jac = -rot @ skew(d)
    analytic = height * (jac[:2, :] / dn[2] - np.outer(dn[:2], jac[2, :]) / dn[2] ** 2)
    numeric = np.zeros((2, 3))
    eps = 1e-7
    for k in range(3):
        dv = np.zeros(3)
        dv[k] = eps
        numeric[:, k] = (offset(quat_product(q, quat_exponential(dv))) -
                         offset(quat_product(q, quat_exponential(-dv)))) / (2 * eps)
    assert np.allclose(analytic, numeric, atol=1e-5)
    # The filter corrects a pitch error from attitude-free ground points.
    f = make(q=(cos(.01), 0., sin(.01), 0.), sigma=default_sigma(s6=.05, s7=.05))
    for i in range(41):
        t = 1_000_000 + 10_000 * i
        f.predict(ImuReading(t, (0., 0., 0.), STILL))
        if i % 10 == 0:
            assert f.update_map_ground_point(MapGroundPointMeasurement(
                t, 0., 0., ((.01, 0.), (0., .01)), 100.)).status.startswith("POSITION")
            assert f.update_map_position(MapPositionMeasurement(
                t, 0., 0., ((.01, 0.), (0., .01)))).status.startswith("POSITION")
    w_, x_, y_, z_ = f.q
    assert abs(np.arcsin(2 * (w_ * y_ - z_ * x_))) < .004
