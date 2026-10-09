"""End-to-end map-candidate reporting, with no automatic state injection."""
import csv
import json

import cv2
import numpy as np
import pytest

from src.run_orthophoto_localization import run_orthophoto_log


def fixtures(tmp_path, *, declared=False):
    rng = np.random.default_rng(920)
    map_img = np.zeros((1100, 1280), np.uint8)
    for _ in range(2500):
        x = int(rng.integers(15, 1265))
        y = int(rng.integers(15, 1085))
        cv2.circle(map_img, (x,y), int(rng.integers(2,5)), 190, -1)
    assert cv2.imwrite(str(tmp_path / "orthophoto.png"), map_img)
    assert cv2.imwrite(str(tmp_path / "frame.png"),
                       map_img[260:740, 240:880])
    config = tmp_path / "orthophoto.yaml"
    config.write_text(
        "camera:\n"
        "  focal_x_px: 1000\n"
        "  focal_y_px: 1000\n"
        "  center_x_px: 320\n"
        "  center_y_px: 240\n"
        "  image_to_body_xy: [[0, -1], [1, 0]]\n"
        "  body_from_camera: [[0, -1, 0], [1, 0, 0], [0, 0, 1]]\n"
        "  camera_offset_body_m: [0, 0, 0]\n"
        "map_tiles:\n"
        "  - name: example_map\n"
        "    image_path: orthophoto.png\n"
        "    origin_ne_m: [100, -50]\n"
        "    matrix_ne_m_per_px: [[0, -0.1], [0.1, 0]]\n"
        "map_covariance_ne_m2: [[0.1, 0], [0, 0.1]]\n"
        f"map_error_statistics_validated: {str(declared).lower()}\n"
        f"measurement_errors_independent_from_prior: {str(declared).lower()}\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "camera.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp_us", "image_path", "height_agl_m",
            "roll_rad", "pitch_rad", "yaw_rad",
            "predicted_n_m", "predicted_e_m",
            "pred_cov_nn_m2", "pred_cov_ne_m2", "pred_cov_ee_m2",
        ])
        writer.writerow([
            1_000_000, "frame.png", 100, 0,0,0,
            50,6, 1.,0.,1.,
        ])
    return manifest, config


def test_offline_map_candidate_without_validated_covariance_is_not_applied(tmp_path):
    log,config=fixtures(tmp_path)
    result=run_orthophoto_log(log,config,tmp_path/"out")
    assert result["frames"]==1
    assert result["geometrically_accepted"]==1
    assert result["applied_to_estimator"]==0
    assert result["correction_gate_counts"]=={"MAP_COVARIANCE_UNVALIDATED":1}
    with (tmp_path/"out"/"orthophoto_candidates.csv").open(newline="") as f:
        row=next(csv.DictReader(f))
    assert float(row["candidate_n_m"])==pytest.approx(50.,abs=.4)
    assert float(row["candidate_e_m"])==pytest.approx(6.,abs=.4)
    assert row["applied_to_estimator"]=="0"
    assert json.loads((tmp_path/"out"/"orthophoto_summary.json").read_text())[
        "applied_to_estimator"
    ]==0


def test_calibrated_independence_declaration_only_allows_candidate_status(tmp_path):
    log,config=fixtures(tmp_path,declared=True)
    report=run_orthophoto_log(log,config,tmp_path/"out")
    assert report["correction_gate_counts"]=={"CANDIDATE_ONLY_NOT_APPLIED":1}
    assert report["applied_to_estimator"]==0


def test_reject_missing_frame_and_out_of_order_timestamps(tmp_path):
    log,config=fixtures(tmp_path)
    (tmp_path/"frame.png").unlink()
    with pytest.raises(ValueError,match="Нет кадра"):
        run_orthophoto_log(log,config,tmp_path/"out")
    log,config=fixtures(tmp_path)
    with log.open(newline="",encoding="utf-8") as f:
        data=list(csv.reader(f))
    data.append(data[1][:])
    with log.open("w",newline="",encoding="utf-8") as f:
        csv.writer(f).writerows(data)
    with pytest.raises(ValueError,match="возрастать"):
        run_orthophoto_log(log,config,tmp_path/"out")
