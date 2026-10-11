"""Ранги наблюдаемости 15-мерного РФК для наборов коррекций."""
from src.eskf_observability import SCENARIOS, observability, report


def by_name():
    return {s.name: s for s in SCENARIOS}


def test_hover_yaw_and_z_gyro_bias_need_heading_aid():
    s = by_name()
    plain = observability(s["hover_body_velocity"], duration_s=8.)
    assert plain["rank"] == 6
    assert {"theta_z", "bg_z", "p_n", "p_e"} <= set(plain["unobservable_states"])
    aided = observability(s["hover_body_velocity_yaw"], duration_s=8.)
    assert "theta_z" not in aided["unobservable_states"]
    assert "bg_z" not in aided["unobservable_states"]
    mapped = observability(s["hover_body_velocity_yaw_map"], duration_s=8.)
    assert "p_n" not in mapped["unobservable_states"]
    # Tilt / accelerometer-bias ambiguity in hover is a known limitation.
    assert {"ba_x", "ba_y"} <= set(mapped["unobservable_states"])


def test_maneuvers_with_body_velocity_and_map_leave_only_vertical_position():
    s = by_name()
    result = observability(s["maneuver_body_velocity_map"], duration_s=8.)
    assert result["rank"] == 14
    assert result["unobservable_states"] == ["p_d"]
    ne_only = observability(s["maneuver_velocity_ne"], duration_s=8.)
    assert "v_d" in ne_only["unobservable_states"]


def test_report_lists_every_scenario():
    data = report()
    assert [r["scenario"] for r in data["scenarios"]] == [s.name for s in SCENARIOS]
