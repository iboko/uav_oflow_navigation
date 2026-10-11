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
