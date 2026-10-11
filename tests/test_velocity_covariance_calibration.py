"""Калибровка ковариации скорости VO по независимому эталону."""
import csv
import json
import subprocess
import sys

import numpy as np
import pytest

from src.velocity_covariance_calibration import (
    VelocityErrorFlight, calibrate_velocity_covariance,
)


def flight(name, rng, n=200, true_scale=2.5, bias=(.02, -.01)):
    model = np.array([[.01, .002], [.002, .02]])
    errors = rng.multivariate_normal(bias, true_scale * model, size=n)
    return VelocityErrorFlight(
        name, tuple(map(tuple, errors)),
        tuple((tuple(model[0]), tuple(model[1])) for _ in range(n)))


def test_scale_and_bias_recovered_and_holdout_consistent():
    rng = np.random.default_rng(5)
    fit = [flight(f"fit{i}", rng) for i in range(4)]
    holdout = [flight(f"hold{i}", rng) for i in range(3)]
    cal = calibrate_velocity_covariance(fit, holdout)
    assert cal.scale == pytest.approx(2.5, rel=.1)
    assert np.allclose(cal.bias_m_s, (.02, -.01), atol=.01)
    assert cal.consistent_on_holdout
    assert 1.6 < cal.holdout_mean_nees_2d < 2.4
    assert cal.independently_validated_for_flight is False
    model = np.array([[.01, .002], [.002, .02]])
    assert np.allclose(cal.covariance(model), cal.scale * model)


def test_optimistic_model_on_holdout_is_flagged():
    rng = np.random.default_rng(6)
    fit = [flight(f"fit{i}", rng, true_scale=1.) for i in range(3)]
    holdout = [flight(f"hold{i}", rng, true_scale=6.) for i in range(2)]
    cal = calibrate_velocity_covariance(fit, holdout)
    assert not cal.consistent_on_holdout
    assert cal.holdout_mean_nees_2d > 4.


def test_overlap_and_too_few_flights_rejected():
    rng = np.random.default_rng(7)
    a = flight("same", rng)
    with pytest.raises(ValueError):
        calibrate_velocity_covariance([a, flight("b", rng), flight("c", rng)],
                                      [a, flight("d", rng)])
    with pytest.raises(ValueError):
        calibrate_velocity_covariance([flight("x", rng)], [flight("y", rng)])


def test_command_line_reads_vo_and_reference(tmp_path):
    rng = np.random.default_rng(8)
    items = {}
    for name in ("f1", "f2", "f3", "h1", "h2"):
        vo, ref = tmp_path / f"{name}_vo.csv", tmp_path / f"{name}_ref.csv"
        with vo.open("w", newline="") as a, ref.open("w", newline="") as b:
            wa, wb = csv.writer(a), csv.writer(b)
            wa.writerow(["timestamp_us", "accepted", "vn_m_s", "ve_m_s", "cov_nn", "cov_ne", "cov_ee"])
            wb.writerow(["timestamp_us", "vn_m_s", "ve_m_s"])
            for k in range(40):
                t = 1_000_000 + 100_000 * k
                e = rng.normal(0., .1, 2)
                wa.writerow([t + 1000, 1, 1. + e[0], .2 + e[1], .01, 0., .01])
                wb.writerow([t, 1., .2])
        items[name] = f"{name}:{vo}:{ref}"
    out = tmp_path / "cal.json"
    subprocess.run([sys.executable, "-m", "src.velocity_covariance_calibration",
                    "--fit", items["f1"], items["f2"], items["f3"],
                    "--holdout", items["h1"], items["h2"],
                    "--output", str(out)], check=True, capture_output=True)
    data = json.loads(out.read_text())
    assert data["scale"] == pytest.approx(1., rel=.35)
    assert set(data["holdout_flights"]) == {"h1", "h2"}
