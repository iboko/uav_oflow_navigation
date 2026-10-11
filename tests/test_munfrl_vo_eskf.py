"""VO + РФК на срезах MUN-FRL: выставка, восстановление меток ИИМ, прогон."""
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
_SPEC = importlib.util.spec_from_file_location("munfrl_vo_eskf", TOOLS / "munfrl_vo_eskf.py")
tool = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tool)


def test_procrustes_recovers_rotation_and_scale():
    rng = np.random.default_rng(1)
    src = rng.normal(0, 3, (300, 2))
    dst = 1.3 * (tool.rot(math.radians(40.)) @ src.T).T + rng.normal(0, .2, (300, 2))
    theta, scale = tool.procrustes(src, dst)
    assert math.degrees(theta) == pytest.approx(40., abs=.5)
    assert scale == pytest.approx(1.3, rel=.01)


def test_dejitter_restores_uniform_grid_from_burst_stamps():
    rng = np.random.default_rng(2)
    true_t = 1_000_000 + np.arange(800) * 2500.
    stamps = true_t + 3000. + rng.uniform(0, 25000., len(true_t))
    fitted, info = tool.dejitter(stamps)
    assert info["rate_hz"] == pytest.approx(400., rel=1e-3)
    assert np.max(np.abs(fitted - (true_t + 3000.))) < 1500.


def _frames_and_imu(velocity=(2.0, 0.0), seconds=8.0):
    """Level flight at constant NED velocity; VO exact, PPK reference exact."""
    imu, frames = [], []
    for k in range(1, int(seconds * 100) + 1):
        imu.append(tool.ImuReading(1_000_000 + k * 10_000, (0., 0., 0.), (0., 0., -9.80665)))
    prev = None
    for k in range(1, int(seconds * 10)):
        t = 1_000_000 + k * 100_000
        frames.append({"t": t, "t_prev": prev, "accepted": prev is not None, "vo": velocity,
                       "h": 30.0, "ref_pos": (velocity[0] * k * .1, velocity[1] * k * .1),
                       "ref_vel": velocity, "ppk_fix": True})
        prev = t
    return imu, frames


def test_vo_aided_filter_tracks_velocity_while_imu_only_does_not():
    imu, frames = _frames_and_imu()
    q0 = (1., 0., 0., 0.)
    aided = tool.run_mode("vo", imu, frames, q0, 0.0225, 1.0)
    assert aided["statuses"].get("VISUAL_CORRECTED", 0) > 60
    v_end = np.array(aided["trajectory"][-1]["v"])
    assert np.allclose(v_end, (2.0, 0.0), atol=0.15)
    metrics = tool.evaluate(frames, aided["trajectory"], 0.0)
    assert metrics["velocity_rmse_m_s"] < 0.2
    unaided = tool.run_mode("imu", imu, frames, q0, 0.0225, 1.0)
    # Starts at rest: the unaided filter never learns the 2 m/s velocity.
    assert np.linalg.norm(np.array(unaided["trajectory"][-1]["v"]) - (2.0, 0.0)) > 1.5


def test_unknown_mode_is_refused():
    imu, frames = _frames_and_imu(seconds=1.0)
    with pytest.raises(ValueError):
        tool.run_mode("gnss", imu, frames, (1., 0., 0., 0.), 0.02, 1.0)
