"""Бортовая привязка к карте: захват, NIS, распространение по VO, отказ карты."""
import numpy as np
import pytest

from src.companion_map import MapAnchor

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
