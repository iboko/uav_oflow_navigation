"""Temporal map consistency is only a screening tool, never a fusion permit."""
from math import isnan
import pytest

from src.map_temporal_integrity import TemporalMapIntegrityMonitor
from src.orthophoto_localization import MapLocalizationFix


def accepted(n, e=0.):
    return MapLocalizationFix(True, "GEOMETRICALLY_ACCEPTED_UNVALIDATED",
                              "map", n, e)


def test_temporal_confirmations_do_not_turn_into_eskf_authorization():
    monitor = TemporalMapIntegrityMonitor(
        max_horizontal_speed_m_s=5.,
        minimum_consistent_intervals=3,
    )
    statuses = []
    for i in range(5):
        x = monitor.inspect(1_000_000 + i*100_000, accepted(i*0.2))
        statuses.append(x.status)
        assert not x.usable_as_eskf_observation
    assert statuses == [
        "MAP_HISTORY_INITIALIZED",
        "MAP_TEMPORAL_CONFIRMATION_PENDING",
        "MAP_TEMPORAL_CONFIRMATION_PENDING",
        "MAP_TEMPORALLY_CONSISTENT_UNVALIDATED",
        "MAP_TEMPORALLY_CONSISTENT_UNVALIDATED",
    ]
    assert x.confirmed_intervals == 4


def test_kinematic_position_jump_latches_until_explicit_reset():
    m = TemporalMapIntegrityMonitor(max_horizontal_speed_m_s=4.)
    m.inspect(1_000_000, accepted(0))
    failed = m.inspect(1_100_000, accepted(15))
    assert failed.status == "MAP_KINEMATIC_INCONSISTENCY"
    assert failed.apparent_speed_m_s == pytest.approx(150.)
    assert m.inspect(1_200_000, accepted(0)).status == "MAP_INTEGRITY_LATCHED"
    m.reset()
    assert m.inspect(1_200_000, accepted(0)).status == "MAP_HISTORY_INITIALIZED"


def test_missing_camera_frame_fails_closed_and_requires_reset():
    m = TemporalMapIntegrityMonitor()
    m.inspect(1_000_000, accepted(0))
    invalid = MapLocalizationFix(False, "INSUFFICIENT_TEXTURE")
    assert m.inspect(1_100_000, invalid).status == "MAP_OBSERVATION_INTERRUPTED"
    assert m.inspect(1_200_000, accepted(0)).status == "MAP_INTEGRITY_LATCHED"


def test_gap_and_out_of_order_timestamps_do_not_bridge():
    m = TemporalMapIntegrityMonitor(max_interframe_gap_s=.12)
    m.inspect(1_000_000, accepted(0))
    assert m.inspect(1_150_000, accepted(0)).status == "MAP_TIME_GAP"
    m.reset()
    m.inspect(1_000_000, accepted(0))
    assert m.inspect(1_000_000, accepted(0)).status == "MAP_TIMESTAMP_OUT_OF_ORDER"


def test_stationary_repeated_false_fix_could_pass_temporal_gate_and_is_never_fused():
    # Time consistency does not prove uniqueness: consistently wrong
    # matches will look kinematically plausible. The status is only a flag.
    m = TemporalMapIntegrityMonitor(minimum_consistent_intervals=2)
    x = None
    for i in range(5):
        x = m.inspect(1_000_000+i*100_000, accepted(900., -700.))
    assert x is not None
    assert x.consistent
    assert x.status == "MAP_TEMPORALLY_CONSISTENT_UNVALIDATED"
    assert x.usable_as_eskf_observation is False
