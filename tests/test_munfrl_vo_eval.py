"""Регрессия временного разделения и входных данных MUN-FRL."""
import csv

import numpy as np
import pytest

from tools.munfrl_vo_eval import evaluate, fit_nonnegative_variances, load


def make_rows(tmp_path, *, q_bad=False):
    manifest, vo = tmp_path / "manifest.csv", tmp_path / "vo.csv"
    with manifest.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp_us", "true_n_m", "true_e_m", "height_agl_m", "ppk_quality"])
        for k in range(32):
            w.writerow([1_000_000 + k * 100_000, k * .5, 0., 35., 0 if q_bad and k == 8 else 1])
    with vo.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp_us", "accepted", "vn_m_s", "ve_m_s"])
        for k in range(1, 32):
            w.writerow([1_000_000 + k * 100_000, 1, 5.1, .1])
    return manifest, vo


def test_temporal_split_does_not_reuse_boundary_frame(tmp_path):
    rows = load(*make_rows(tmp_path))
    report = evaluate(rows, 1.5)
    assert report["boundary_frames_shared"] == 0
    assert report["excluded_boundary_intervals"] >= 1
    assert report["calibration"]["after"]["intervals"] >= 5
    assert report["validation"]["with_profile_constants"]["intervals"] >= 5
    assert report["independent_flight_validation"] is False
    assert report["covariance_applied_to_px4"] is False


def test_temporal_gap_excludes_more_intervals(tmp_path):
    rows = load(*make_rows(tmp_path))
    short = evaluate(rows, 1.5)
    long = evaluate(rows, 1.5, .3)
    assert long["excluded_boundary_intervals"] > short["excluded_boundary_intervals"]


def test_nonfixed_ppk_pairs_are_excluded(tmp_path):
    paths = make_rows(tmp_path, q_bad=True)
    assert len(load(*paths)) == 29
    assert len(load(*paths, require_ppk_fix=False)) == 31


def test_invalid_ppk_quality_header_is_rejected(tmp_path):
    m, v = make_rows(tmp_path)
    m.write_text(m.read_text().replace(",ppk_quality", ",quality"))
    with pytest.raises(ValueError, match="ppk_quality"):
        load(m, v)


def test_duplicate_accepted_vo_frame_is_rejected(tmp_path):
    m, v = make_rows(tmp_path)
    with v.open("a") as f:
        f.write("1100000,1,5.1,0.1\n")
    with pytest.raises(ValueError, match="Дублирующийся"):
        load(m, v)


def test_nnls_matches_global_optimum_from_nonnegative_candidates():
    matrix = np.array([[1., 1., 0.], [1., 2., 1.],
                       [1., 3., 4.], [1., 4., 9.]], dtype=float)
    target = np.array([2., 1., 1., 2.])
    fit = fit_nonnegative_variances(matrix, target)
    assert np.all(fit >= 0)
    # Grid-based independent regression oracle, accurate to coarse tolerance.
    alternatives = [
        np.array([a, b, c])
        for a in np.arange(0, 2.1, .1)
        for b in np.arange(0, 1.1, .1)
        for c in np.arange(0, .51, .1)
    ]
    best_grid = min(float(np.sum((matrix @ x - target) ** 2)) for x in alternatives)
    assert np.sum((matrix @ fit - target) ** 2) <= best_grid + 1e-9


def test_fitted_noise_model_not_identifiable_with_constant_height_and_speed(tmp_path):
    rows = load(*make_rows(tmp_path))
    report = evaluate(rows, 1.5)
    assert report["calibration"]["noise_model_design_rank"] == 1
    assert report["calibration"]["noise_model_identifiable"] is False
    assert report["calibration"]["fitted_noise_model"] is None
    assert "mean_nees" not in report["validation"]["with_fitted_model"]
    assert report["calibration"]["noise_model_warning"]
    assert report["calibration"]["fitted_noise_model_validated_for_flight"] is False


def test_noise_model_identifiability_with_distinct_height_and_speed_excitation(tmp_path):
    rows = load(*make_rows(tmp_path))
    for k, row in enumerate(rows):
        row["h"] = 32.0 + .3 * k + 2.0 * np.sin(k / 3.)
        row["ref"] = np.array([3.0 + .1 * k, np.sin(.4 * k)])
        row["vo"] = row["ref"] * 1.1 + np.array([.01 * np.sin(k), .02 * np.cos(k)])
    report = evaluate(rows, 1.5)
    assert report["calibration"]["noise_model_design_rank"] == 3
    assert report["calibration"]["noise_model_identifiable"] is True
    assert report["calibration"]["noise_model_normalized_condition_number"] < 1e4
    assert report["calibration"]["fitted_noise_model_validated_for_flight"] is False
