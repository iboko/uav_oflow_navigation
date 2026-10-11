"""MSCKF компаньона: выставка с дальномером, затем скорость в осях корпуса."""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import test_msckf_vio as sim  # noqa: E402

from src.companion_msckf import CompanionMsckf  # noqa: E402
from src.imu_preintegration import rotation_body_to_ned  # noqa: E402


def _rp(q):
    w, x, y, z = q
    return (math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
            math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x)))))


def test_companion_msckf_initialises_from_rangefinder_and_tracks_body_velocity():
    imu, frames, truth = sim._simulate(seconds=10.0)
    comp = CompanionMsckf(sim.CAM, sim.NOISE)
    outs = []
    k = 0
    for (t_us, feats), (_, p, v) in zip(frames, truth):
        while k < len(imu) and imu[k].timestamp_us <= t_us + 20_000:  # IMU arrives a little ahead
            comp.add_imu(imu[k])
            k += 1
        q = sim._truth(t_us * 1e-6)[3]
        roll, pitch = _rp(q)
        rng = -p[2] / (math.cos(roll) * math.cos(pitch))  # rangefinder along body z, flat ground
        out = comp.process(t_us, feats, attitude_q=tuple(q), roll=roll, pitch=pitch, range_m=rng)
        if out is not None:
            v_b_true = rotation_body_to_ned(q).T @ v
            outs.append(np.subtract(out.velocity_frd, v_b_true))
    assert comp.counts.get("INIT_INITIALIZED") == 1, comp.counts
    assert len(outs) > 60
    err = np.array(outs)
    assert np.sqrt(np.mean(np.sum(err ** 2, axis=1))) < 0.15


def test_companion_msckf_waits_below_minimum_height():
    imu, frames, _ = sim._simulate(seconds=3.0)
    comp = CompanionMsckf(sim.CAM, sim.NOISE)
    for r in imu:
        comp.add_imu(r)
    for t_us, feats in frames:
        assert comp.process(t_us, feats, attitude_q=(1.0, 0, 0, 0), roll=0.0, pitch=0.0, range_m=1.0) is None
    assert comp.counts["BELOW_MIN_HEIGHT"] == len(frames)


def _run_with_dropped_imu(drop):
    imu, frames, truth = sim._simulate(seconds=10.0)
    comp = CompanionMsckf(sim.CAM, sim.NOISE)
    kept = [r for r in imu if not drop(r.timestamp_us)]
    outs, k = [], 0
    for (t_us, feats), (_, p, v) in zip(frames, truth):
        while k < len(kept) and kept[k].timestamp_us <= t_us + 20_000:
            comp.add_imu(kept[k])
            k += 1
        q = sim._truth(t_us * 1e-6)[3]
        roll, pitch = _rp(q)
        rng = -p[2] / (math.cos(roll) * math.cos(pitch))
        out = comp.process(t_us, feats, attitude_q=tuple(q), roll=roll, pitch=pitch, range_m=rng)
        if out is not None:
            outs.append(np.subtract(out.velocity_frd, rotation_body_to_ned(q).T @ v))
    return comp, np.array(outs)


def test_companion_msckf_bridges_short_imu_dropouts():
    # Lost MAVLink packets: ~50 ms holes every second (seen in SITL).
    comp, err = _run_with_dropped_imu(lambda t: 4_000_000 < t and (t % 1_000_000) < 45_000)
    assert comp.counts.get("IMU_GAP_BRIDGED", 0) >= 5, comp.counts
    assert "RESET_IMU_GAP" not in comp.counts and "RESET_NUMERICAL" not in comp.counts, comp.counts
    assert len(err) > 60
    assert np.sqrt(np.mean(np.sum(err ** 2, axis=1))) < 0.15


def test_companion_msckf_resets_on_long_imu_loss():
    comp, _ = _run_with_dropped_imu(lambda t: 5_000_000 < t < 5_300_000)
    assert comp.counts.get("RESET_IMU_GAP") == 1, comp.counts
    assert comp.counts.get("INIT_INITIALIZED", 0) >= 2, comp.counts
