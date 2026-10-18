"""Pure helpers of the flight replay video (no ULog, no ffmpeg)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

GEOREF = {"origin_ne_m": [259.83, -259.83], "matrix_ne_m_per_px": [[0.0, -0.25], [0.25, 0.0]]}


def test_ne_to_px_inverts_georef():
    import flight_replay_video as fr
    px = np.array([[100.0, 40.0], [1040.0, 1040.0]])
    a, o = np.asarray(GEOREF["matrix_ne_m_per_px"]), np.asarray(GEOREF["origin_ne_m"])
    ne = px @ a.T + o
    assert np.allclose(fr.ne_to_px(ne, GEOREF), px)
    # North is up (v decreases), east is right (u increases).
    (u0, v0), (u1, v1) = fr.ne_to_px([(0.0, 0.0), (1.0, 1.0)], GEOREF)
    assert v1 < v0 and u1 > u0


def test_view_box_square_with_margin():
    import flight_replay_video as fr
    n0, n1, e0, e1 = fr.view_box([(0.0, 0.0), (-40.0, 10.0)], margin_m=5.0, min_size_m=20.0)
    assert np.isclose(n1 - n0, e1 - e0) and np.isclose(n1 - n0, 50.0)
    assert n0 <= -45.0 and n1 >= 5.0 and e0 <= -5.0 and e1 >= 15.0
    n0, n1, _, _ = fr.view_box([(1.0, 1.0)], margin_m=1.0, min_size_m=60.0)
    assert np.isclose(n1 - n0, 60.0)


def test_phase_at_follows_events():
    import flight_replay_video as fr
    rep = {"takeoff_us": 10, "gnss_off_us": 20, "route_end_us": 30, "fault_start_us": 40, "fault_end_us": 50}
    assert fr.phase_at(5, rep) == "pre-flight"
    assert "map lock" in fr.phase_at(15, rep)
    assert "GNSS OFF" in fr.phase_at(25, rep)
    assert "FAULT" in fr.phase_at(45, rep)
    assert "hold" in fr.phase_at(35, rep) and "hold" in fr.phase_at(55, rep)
    assert "GNSS OFF" in fr.phase_at(25, {"takeoff_us": 10, "gnss_off_us": 20})


def test_phase_at_calibration_flight():
    import flight_replay_video as fr
    rep = {"type": "PX4_SITL_GAZEBO_COMPANION_MAP_CALIBRATE", "takeoff_us": 10}
    assert fr.phase_at(5, rep) == "pre-flight" and "calibration" in fr.phase_at(15, rep)
