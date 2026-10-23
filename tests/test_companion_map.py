"""Бортовая привязка к карте: захват, NIS, распространение по VO, отказ карты."""
import numpy as np
import pytest

from src.companion_map import AnchorMeasurement, GyroHeading, MapAnchor, VelocityBridge

R = ((0.05 ** 2, 0.0), (0.0, 0.05 ** 2))
VO_COV = ((0.05 ** 2, 0.0), (0.0, 0.05 ** 2))
DT = 66_000


def _fly(anchor, n, v=(1.0, 0.0), start=(0.0, 0.0), t0=1_000_000, map_shift=(0.0, 0.0),
         vo_scale=1.0, fix_every=1, heading_error=0.0):
    pos = np.array(start, dtype=float)
    t = t0
    results = []
    for k in range(n):
        # VO in N/E with the autopilot heading, which is off by heading_error.
        c, s = np.cos(-heading_error), np.sin(-heading_error)
        v_vo = np.array([[c, -s], [s, c]]) @ np.array(v) * vo_scale
        anchor.propagate(t, t + DT, tuple(v_vo), VO_COV)
        t += DT
        pos = pos + np.array(v) * DT * 1e-6
        if k % fix_every == 0:
            results.append(anchor.fix(t, True, pos[0] + map_shift[0], pos[1] + map_shift[1],
                                      heading_error))
    return results, pos, t


def test_lock_needs_kinematically_consistent_fixes():
    anchor = MapAnchor(bias_ne_m=(0.0, 0.0), covariance_ne_m2=R)
    results, pos, _ = _fly(anchor, 3)
    assert results == ["ACQUISITION_PENDING", "ACQUISITION_PENDING", "LOCK_ACQUIRED"]
    assert np.allclose(anchor.status().position_ne_m, pos)


def test_acquisition_restarts_when_map_jumps_against_vo():
    anchor = MapAnchor(bias_ne_m=(0.0, 0.0), covariance_ne_m2=R)
    anchor.propagate(1_000_000, 1_066_000, (0.0, 0.0), VO_COV)
    assert anchor.fix(1_066_000, True, 0.0, 0.0) == "ACQUISITION_PENDING"
    anchor.propagate(1_066_000, 1_132_000, (0.0, 0.0), VO_COV)
    # Карта «прыгнула» на 2 м, VO говорит — стоим.
    assert anchor.fix(1_132_000, True, 2.0, 0.0) == "ACQUISITION_INCONSISTENT"
    assert not anchor.locked


def test_vo_scale_error_is_corrected_by_map_fixes():
    anchor = MapAnchor(bias_ne_m=(0.0, 0.0), covariance_ne_m2=R)
    results, pos, _ = _fly(anchor, 150, vo_scale=1.05, fix_every=8)
    assert "FIX_FUSED" in results and "FIX_REJECTED_NIS" not in results
    err = np.linalg.norm(np.array(anchor.status().position_ne_m) - pos)
    assert err < 0.15  # только VO: 5 % от 9,9 м пути = 0,5 м


def test_shifted_map_is_rejected_after_lock_and_position_continues_on_vo():
    anchor = MapAnchor(bias_ne_m=(0.0, 0.0), covariance_ne_m2=R)
    _, pos, t = _fly(anchor, 10)
    results, pos2, t = _fly(anchor, 30, start=pos, t0=t, map_shift=(0.0, 3.0), fix_every=5)
    assert set(results) == {"FIX_REJECTED_NIS"}
    assert anchor.locked
    assert np.linalg.norm(np.array(anchor.status().position_ne_m) - pos2) < 0.1
    results, _, _ = _fly(anchor, 10, start=pos2, t0=t)
    assert results[0] == "FIX_FUSED"


def test_calibrated_bias_is_removed():
    anchor = MapAnchor(bias_ne_m=(0.2, -0.1), covariance_ne_m2=R)
    _fly(anchor, 3, map_shift=(0.2, -0.1))
    assert anchor.status().position_ne_m == pytest.approx(
        (3 * DT * 1e-6, 0.0), abs=1e-9)


def test_vo_gap_inflates_covariance_and_long_outage_drops_lock():
    anchor = MapAnchor(bias_ne_m=(0.0, 0.0), covariance_ne_m2=R, max_sigma_m=1.0)
    _, _, t = _fly(anchor, 3)
    var0 = anchor.status().covariance_ne_m2[0][0]
    anchor.propagate(t + 200_000, t + 266_000, (0.0, 0.0), VO_COV)  # 0,2 с без VO
    var1 = anchor.status().covariance_ne_m2[0][0]
    assert var1 == pytest.approx(var0 + 0.05 ** 2 * 0.066 ** 2 + (2.0 * 0.2) ** 2, rel=1e-6)
    assert anchor.fix(t + 200_000, True, 0.0, 0.0) == "FIX_TIME_MISMATCH"
    anchor.propagate(t + 266_000 + 600_000, t + 932_000, (0.0, 0.0), VO_COV)
    assert not anchor.locked and anchor.counts["LOCK_LOST_UNCERTAINTY"] == 1


def test_invalid_calibration_is_refused():
    with pytest.raises(ValueError):
        MapAnchor(bias_ne_m=(0.0, 0.0), covariance_ne_m2=((0.0, 0.0), (0.0, 0.0)))
    with pytest.raises(ValueError):
        MapAnchor(bias_ne_m=(float("nan"), 0.0), covariance_ne_m2=R)


def test_heading_offset_from_map_fixes_corrects_vo_direction():
    anchor = MapAnchor(bias_ne_m=(0.0, 0.0), covariance_ne_m2=R)
    results, pos, _ = _fly(anchor, 300, fix_every=8, heading_error=np.radians(9.0))
    assert anchor.heading_offset_rad == pytest.approx(np.radians(9.0), abs=1e-6)
    assert "FIX_REJECTED_NIS" not in results
    # Long gap without fixes: VO alone, but rotated by the measured offset.
    _, pos2, _ = _fly(anchor, 60, start=pos, t0=1_000_000 + 300 * DT, fix_every=1000,
                      heading_error=np.radians(9.0))
    assert np.linalg.norm(np.array(anchor.status().position_ne_m) - pos2) < 0.05


def test_heading_sigma_has_a_floor_and_follows_the_spread():
    anchor = MapAnchor(bias_ne_m=(0.0, 0.0), covariance_ne_m2=R)
    assert not anchor.heading_known
    _fly(anchor, 200, fix_every=4, heading_error=np.radians(9.0))
    assert anchor.heading_known
    assert anchor.heading_sigma_rad() == pytest.approx(0.05)  # identical offsets: floor
    assert anchor.heading_sigma_rad(floor_rad=0.0) < 1e-6


def test_last_fix_time_tracks_only_map_backed_updates():
    anchor = MapAnchor(bias_ne_m=(0.0, 0.0), covariance_ne_m2=R)
    _, pos, t = _fly(anchor, 3)
    assert anchor.last_fix_us == t  # lock
    locked_at = anchor.last_fix_us
    _fly(anchor, 10, start=pos, t0=t, map_shift=(0.0, 3.0), fix_every=5)  # all rejected
    assert anchor.last_fix_us == locked_at


def test_repeated_correlated_fixes_do_not_collapse_covariance():
    anchor = MapAnchor(bias_ne_m=(0.0, 0.0), covariance_ne_m2=R)
    _fly(anchor, 150, v=(0.0, 0.0))
    cov = np.asarray(anchor.status().covariance_ne_m2)
    # Independent-update formula would shrink R nearly 150-fold.
    assert np.linalg.eigvalsh(cov - np.asarray(R)).min() >= -1e-12
    assert anchor.counts["FIX_FUSED"] == 147


def test_wide_covariance_does_not_hide_an_abrupt_shift_or_poison_reference():
    wide = ((0.82 ** 2, 0.0), (0.0, 0.82 ** 2))
    anchor = MapAnchor(bias_ne_m=(0.0, 0.0), covariance_ne_m2=wide)
    _, pos, t = _fly(anchor, 3)
    last_fix = anchor.last_fix_us
    results, pos, t = _fly(anchor, 30, start=pos, t0=t, map_shift=(0.0, 3.0), fix_every=5)
    assert set(results) == {"FIX_REJECTED_STEP"}
    assert anchor.last_nis < 9.21  # marginal gate alone would accept
    assert anchor.last_fix_us == last_fix
    results, _, _ = _fly(anchor, 3, start=pos, t0=t)
    assert set(results) == {"FIX_FUSED"}


def test_step_gate_compares_displacement_not_distance_travelled():
    wide = ((0.82 ** 2, 0.0), (0.0, 0.82 ** 2))
    anchor = MapAnchor(bias_ne_m=(0.0, 0.0), covariance_ne_m2=wide)
    _, pos, t = _fly(anchor, 3)
    # Move 10 m between map fixes: this is not a map jump.
    anchor.propagate(t, t + 1_000_000, (10.0, 0.0), VO_COV)
    assert anchor.fix(t + 1_000_000, True, pos[0] + 10.0, pos[1]) == "FIX_FUSED"


def test_duplicate_map_fix_cannot_refresh_freshness_or_heading():
    anchor = MapAnchor(bias_ne_m=(0.0, 0.0), covariance_ne_m2=R)
    _, pos, t = _fly(anchor, 3)
    before = anchor.status()
    assert anchor.fix(t, True, *pos, heading_offset_rad=1.0) == "FIX_DUPLICATE_OR_OLD"
    assert anchor.last_fix_us == t
    assert anchor.status().position_ne_m == before.position_ne_m
    assert anchor.heading_offset_rad == before.heading_offset_rad


@pytest.mark.parametrize("bad", [((1.0, 2.0), (0.0, 1.0)),
                                  ((-1.0, 0.0), (0.0, 1.0))])
def test_bad_vo_covariance_does_not_change_anchor(bad):
    anchor = MapAnchor(bias_ne_m=(0.0, 0.0), covariance_ne_m2=R)
    _, _, t = _fly(anchor, 3)
    before = anchor.status()
    anchor.propagate(t, t + DT, (1.0, 0.0), bad)
    assert anchor.status().position_ne_m == before.position_ne_m
    assert anchor.status().covariance_ne_m2 == before.covariance_ne_m2


@pytest.mark.parametrize("limit", [0.0, -1.0, float("nan"), float("inf")])
def test_step_limit_must_be_positive_and_finite(limit):
    with pytest.raises(ValueError):
        MapAnchor(bias_ne_m=(0.0, 0.0), covariance_ne_m2=R, max_map_step_error_m=limit)


def test_persistent_scale_uncertainty_is_independent_of_frame_partition():
    covariances = []
    for frames in (1, 100):
        anchor = MapAnchor(bias_ne_m=(0.0, 0.0), covariance_ne_m2=R,
                           heading_sigma_unknown_rad=0.0, heading_sigma_known_rad=0.0,
                           max_sigma_m=10.0)
        _, _, t = _fly(anchor, 3, v=(0.0, 0.0))
        dt_us = 1_000_000 // frames
        for _ in range(frames):
            anchor.propagate(t, t + dt_us, (10.0, 0.0), ((1e-12, 0.0), (0.0, 1e-12)))
            t += dt_us
        covariances.append(np.asarray(anchor.status().covariance_ne_m2))
    # 5% constant scale error on 10 m: 0.5 m sigma, 0.25 m² variance.
    expected = np.asarray(R) + np.eye(2) * 0.25
    for cov in covariances:
        assert np.allclose(cov, expected, rtol=0, atol=2e-12)


def test_long_visual_outage_loses_lock_from_accumulated_scale_uncertainty():
    anchor = MapAnchor(bias_ne_m=(0.0, 0.0), covariance_ne_m2=R,
                       heading_sigma_unknown_rad=0.0, max_sigma_m=0.4)
    _, _, t = _fly(anchor, 3, v=(0.0, 0.0))
    for _ in range(100):
        anchor.propagate(t, t + 10_000, (10.0, 0.0), VO_COV)
        t += 10_000
    # The old sum of squared per-frame scale errors gave sigma ~0.07 m.
    assert not anchor.locked
    assert anchor.counts["LOCK_LOST_UNCERTAINTY"] == 1
    assert anchor.measurement_at(t, max_fix_age_s=10.0) is None


def test_midpoint_position_uses_heading_corrected_velocity():
    anchor = MapAnchor(bias_ne_m=(0.0, 0.0), covariance_ne_m2=R,
                       heading_fixes_needed=1)
    _, pos, t = _fly(anchor, 3, v=(0.0, 0.0), heading_error=np.pi / 2)
    anchor.propagate(t, t + 100_000, (0.0, -2.0), VO_COV)
    measurement = anchor.measurement_at(t + 50_000, max_fix_age_s=1.0)
    assert measurement is not None
    assert measurement.timestamp_us == t + 50_000
    assert measurement.position_ne_m == pytest.approx((pos[0] + 0.1, pos[1]), abs=1e-12)
    # Retiming must not make the position look more precise than the end state.
    delta = np.asarray(measurement.covariance_ne_m2) - np.asarray(anchor.status().covariance_ne_m2)
    assert np.linalg.eigvalsh(delta).min() >= 0


def test_map_position_export_refuses_stale_and_extrapolated_timestamps():
    anchor = MapAnchor(bias_ne_m=(0.0, 0.0), covariance_ne_m2=R)
    _, _, t = _fly(anchor, 3, v=(0.0, 0.0))
    anchor.propagate(t, t + 200_000, (0.0, 0.0), VO_COV)
    # The interval of the locking fix is still stored; before it, nothing.
    assert anchor.measurement_at(t - DT - 1, max_fix_age_s=1.0) is None
    assert anchor.measurement_at(t + 200_001, max_fix_age_s=1.0) is None
    assert anchor.measurement_at(t + 100_000, max_fix_age_s=0.1) is None
    assert anchor.measurement_at(t + 100_000, max_fix_age_s=1.0) is not None
    anchor.last_fix_us = t + 300_000  # corrupt/future reference cannot pass freshness
    assert anchor.measurement_at(t + 200_000, max_fix_age_s=1.0) is None


@pytest.mark.parametrize("age", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_map_freshness_limit_is_refused(age):
    anchor = MapAnchor(bias_ne_m=(0.0, 0.0), covariance_ne_m2=R)
    with pytest.raises(ValueError):
        anchor.measurement_at(0, max_fix_age_s=age)


def test_mavlink_position_covariance_preserves_ne_correlation():
    p = ((0.4, -0.1), (-0.1, 0.3))
    measurement = AnchorMeasurement(1_000_000, (1.0, 2.0), p)
    packed = measurement.mavlink_pose_covariance(0.02)
    matrix = np.zeros((6, 6))
    upper = np.triu_indices(6)
    matrix[upper] = packed
    matrix[(upper[1], upper[0])] = packed
    assert np.allclose(matrix[:2, :2], p)
    assert matrix[2, 2] == 0.02
    assert np.isnan(np.diag(matrix)[3:]).all()


def test_fix_covariance_override_is_used_and_validated():
    anchor = MapAnchor(bias_ne_m=(0.0, 0.0), covariance_ne_m2=R)
    _fly(anchor, 3)
    wide = ((1.0, 0.0), (0.0, 1.0))
    t = anchor.status()
    anchor.propagate(1_198_000, 1_264_000, (1.0, 0.0), VO_COV)
    # 0.5 m off: rejected by NIS with the narrow covariance, accepted with the wide one.
    p = np.asarray(anchor.status().position_ne_m)
    assert anchor.fix(1_264_000, True, p[0] + 0.5, p[1], covariance_ne_m2=wide) == "FIX_FUSED"
    anchor.propagate(1_264_000, 1_330_000, (1.0, 0.0), VO_COV)
    bad = ((1.0, 2.0), (2.0, 1.0))
    assert anchor.fix(1_330_000, True, 0.0, 0.0, covariance_ne_m2=bad) == "FIX_BAD_COVARIANCE"
    assert t.locked


def test_relock_after_loss_rejects_a_self_consistent_false_place():
    anchor = MapAnchor(bias_ne_m=(0.0, 0.0), covariance_ne_m2=R, max_sigma_m=0.5)
    _fly(anchor, 3)                                   # locked
    # No fixes for a while: lock lost by uncertainty, dead reckoning kept.
    t = 1_198_000
    for _ in range(200):
        anchor.propagate(t, t + DT, (1.0, 0.0), VO_COV)
        t += DT
    assert not anchor.locked and anchor.counts.get("LOCK_LOST_UNCERTAINTY") == 1
    truth_n = 0.066 * 203  # 1 m/s for 203 intervals
    # A wrong place 20 m off, consistent with VO from fix to fix: rejected.
    for k in range(4):
        anchor.propagate(t, t + DT, (1.0, 0.0), VO_COV)
        t += DT
        r = anchor.fix(t, True, truth_n + 0.066 * (k + 1) + 20.0, 0.0)
        assert r == "ACQUISITION_REJECTED_PRIOR"
    # The true place: reacquired.
    results = []
    for k in range(4, 7):
        anchor.propagate(t, t + DT, (1.0, 0.0), VO_COV)
        t += DT
        results.append(anchor.fix(t, True, truth_n + 0.066 * (k + 1), 0.0))
    assert results[-1] == "LOCK_ACQUIRED", results


def test_first_lock_is_gated_by_an_independent_prior():
    anchor = MapAnchor(bias_ne_m=(0.0, 0.0), covariance_ne_m2=R)
    anchor.set_acquisition_prior((0.0, 0.0), ((9.0, 0.0), (0.0, 9.0)))
    anchor.propagate(1_000_000, 1_066_000, (0.0, 0.0), VO_COV)
    assert anchor.fix(1_066_000, True, 15.0, 0.0) == "ACQUISITION_REJECTED_PRIOR"
    anchor.propagate(1_066_000, 1_132_000, (0.0, 0.0), VO_COV)
    assert anchor.fix(1_132_000, True, 1.0, 0.0) == "ACQUISITION_PENDING"


def test_gyro_heading_integrates_body_rates_and_ignores_attitude_yaw():
    g = GyroHeading()
    t = 1_000_000
    # Level, 0.1 rad/s for 10 s; the attitude yaw is jumped by an "EKF2 reset"
    # in the middle and must not enter the integrated heading.
    for k in range(1001):
        yaw = 0.3 + (0.5 if k > 500 else 0.0)
        g.add(t + k * 10_000, 0.0, 0.0, yaw, (0.0, 0.0, 0.1))
    assert g.at(t + 10_000_000) == pytest.approx(0.3 + 1.0, abs=1e-6)
    assert np.isnan(g.at(t + 10_000_001)) and np.isnan(g.at(t - 1))
    # Banked turn: yaw rate = (q sin(phi) + r cos(phi)) / cos(theta).
    g2 = GyroHeading()
    phi = np.radians(20.0)
    for k in range(101):
        g2.add(t + k * 10_000, phi, 0.0, 0.0, (0.0, 0.1 * np.sin(phi), 0.1 * np.cos(phi)))
    assert g2.at(t + 1_000_000) == pytest.approx(0.1, abs=1e-6)
    g2.add(t + 2_000_000, phi, 0.0, 0.0, (0.0, 0.1 * np.sin(phi), 0.1 * np.cos(phi)))
    assert g2.gaps == 1


def _heading_loop(use_gyro: bool, steps: int = 400) -> float:
    """EKF2 heading pulled towards the yaw we send (EV yaw); true heading 0,
    autopilot starts 10 deg off, map fixes every 8 intervals measure 0."""
    anchor = MapAnchor(bias_ne_m=(0.0, 0.0), covariance_ne_m2=R)
    psi_ap = psi_g = np.radians(10.0)       # gyro heading starts at the autopilot heading
    t = 1_000_000
    for k in range(steps):
        ref = psi_g - psi_ap if use_gyro else 0.0
        anchor.propagate(t, t + DT, (0.0, 0.0), VO_COV, reference_minus_autopilot_rad=ref)
        t += DT
        if k % 8 == 0:
            anchor.fix(t, True, 0.0, 0.0, 0.0 - (psi_g if use_gyro else psi_ap))
        if anchor.heading_known:
            sent = (psi_g if use_gyro else psi_ap) + anchor.heading_offset_rad
            psi_ap += 0.3 * np.arctan2(np.sin(sent - psi_ap), np.cos(sent - psi_ap))
    return float(np.degrees(psi_ap))


def test_gyro_reference_breaks_the_ev_yaw_feedback_loop():
    # Offset against the autopilot heading: applied again after EKF2 has
    # already turned, the heading runs past the truth (test M: 48 deg).
    assert abs(_heading_loop(use_gyro=False)) > 2.0
    assert abs(_heading_loop(use_gyro=True)) < 0.1


def test_window_midpoint_position_from_an_earlier_interval():
    anchor = MapAnchor(bias_ne_m=(0.0, 0.0), covariance_ne_m2=R, heading_sigma_unknown_rad=0.0)
    _, pos, t = _fly(anchor, 3, v=(0.0, 0.0))
    start = t
    for _ in range(4):
        anchor.propagate(t, t + 100_000, (2.0, 0.0), VO_COV)
        t += 100_000
    # Midpoint of a 4-interval window: inside the second interval.
    m = anchor.measurement_at(start + 150_000, max_fix_age_s=10.0)
    assert m is not None
    assert m.position_ne_m == pytest.approx((pos[0] + 0.3, pos[1]), abs=1e-9)
    # The state of that time, not the later (larger) end covariance.
    assert np.trace(m.covariance_ne_m2) < np.trace(anchor.status().covariance_ne_m2)
    # Before the lock-time interval: nothing stored.
    assert anchor.measurement_at(start - DT - 1, max_fix_age_s=10.0) is None
    # A VO gap clears the history: no position across it.
    anchor.propagate(t + 50_000, t + 150_000, (2.0, 0.0), VO_COV)
    assert anchor.measurement_at(start + 150_000, max_fix_age_s=10.0) is None


def test_bridged_vo_gap_keeps_the_lock_and_rejects_a_shift_behind_it():
    # Strong gusts (series VT, seed 8): a 4.7-s VO gap inside the map fault.
    # Bound on unknown motion: 2 m/s x 4.7 s -> lock lost, then locked again
    # on the shifted map. Bridged by the autopilot velocity it holds.
    for bridged in (False, True):
        anchor = MapAnchor(bias_ne_m=(0.0, 0.0), covariance_ne_m2=R)
        _, pos, t = _fly(anchor, 5)
        gap = 4_700_000
        kw = {"gap_displacement_ne": (4.7, 0.0), "gap_sigma_m": 0.3} if bridged else {}
        anchor.propagate(t + gap, t + gap + DT, (1.0, 0.0), VO_COV, **kw)
        t += gap + DT
        pos = pos + np.array([4.7 + DT * 1e-6, 0.0])
        if not bridged:
            assert not anchor.locked and anchor.counts["LOCK_LOST_UNCERTAINTY"] == 1
            continue
        assert anchor.locked and anchor.counts["VO_GAP_BRIDGED"] == 1
        assert np.allclose(anchor.status().position_ne_m, pos)
        assert anchor.fix(t, True, pos[0], pos[1] + 3.0) in ("FIX_REJECTED_STEP", "FIX_REJECTED_NIS")
        anchor.propagate(t, t + DT, (1.0, 0.0), VO_COV)
        t += DT
        assert anchor.fix(t, True, pos[0] + DT * 1e-6, pos[1]) == "FIX_FUSED"


def test_bridged_vo_gap_keeps_acquisition_candidates():
    anchor = MapAnchor(bias_ne_m=(0.0, 0.0), covariance_ne_m2=R)
    _, pos, t = _fly(anchor, 2)                     # two candidates
    anchor.propagate(t + 2_000_000, t + 2_000_000 + DT, (1.0, 0.0), VO_COV,
                     gap_displacement_ne=(2.0, 0.0), gap_sigma_m=0.2)
    t += 2_000_000 + DT
    assert anchor.fix(t, True, pos[0] + 2.0 + DT * 1e-6, pos[1]) == "LOCK_ACQUIRED"


def test_unbridged_vo_gap_restarts_acquisition():
    anchor = MapAnchor(bias_ne_m=(0.0, 0.0), covariance_ne_m2=R)
    _, pos, t = _fly(anchor, 2)
    anchor.propagate(t + 2_000_000, t + 2_000_000 + DT, (1.0, 0.0), VO_COV)
    assert anchor.counts["ACQUISITION_VO_GAP"] == 1


def test_velocity_bridge_integrates_and_needs_dense_coverage():
    b = VelocityBridge(0.1, 0.05, 0.01)
    for k in range(0, 101):
        b.add(k * 20_000, 1.0 + 0.01 * k, -0.5)     # 50 Hz, 2 s
    d, sigma = b.displacement(100_000, 1_900_000)
    # v(t) = 1 + 0.5 t: integral over [0.1, 1.9] s.
    assert d[0] == pytest.approx(1.8 + 0.25 * (1.9 ** 2 - 0.1 ** 2), rel=1e-6)
    assert d[1] == pytest.approx(-0.9)
    assert sigma == pytest.approx(np.sqrt(0.1 ** 2 + (0.05 * 1.8) ** 2 + (0.01 * 1.8 ** 2) ** 2))
    assert b.displacement(1_900_000, 2_100_000) is None   # beyond the history
    sparse = VelocityBridge(0.1, 0.05, 0.01)
    sparse.add(0, 1.0, 0.0)
    sparse.add(1_000_000, 1.0, 0.0)
    assert sparse.displacement(0, 1_000_000) is None
    with pytest.raises(ValueError):
        VelocityBridge(-0.1, 0.0, 0.0)
