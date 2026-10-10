"""Reproducible synthetic surrogate, never presented as NII VK flight data."""
import json
import numpy as np
import pytest

from src.synthetic_map_campaign import CampaignConfig, run_campaign


@pytest.fixture(scope="module")
def campaign():
    return run_campaign(CampaignConfig(seed=14821, frames_per_flight=60))


def test_synthetic_campaign_has_no_real_flight_validation(campaign):
    assert campaign["campaign_type"] == "SYNTHETIC_STATISTICAL_SURROGATE_ONLY"
    assert not campaign["passed_flight_validation"]
    assert not campaign["px4_eskf_state_mutated"]
    assert set(campaign["scenarios"]) == {
        "nominal", "shared_correlation", "false_match_burst", "camera_outage"
    }
    for data in campaign["scenarios"].values():
        assert set(data["calibration_train_flights"]).isdisjoint(
            data["calibration_holdout_flights"]
        )
        assert data["covariance_source"] == "SYNTHETIC_TRAINING_NOT_INDEPENDENTLY_VALIDATED"
        assert data["total_holdout_frames"] == 180
        assert data["prior"]["evaluated_frames"] > 0
        assert data["map_bias_corrected"]["evaluated_frames"] > 0


def test_injected_map_outlier_and_blackout_are_detectable(campaign):
    bursts = campaign["scenarios"]["false_match_burst"]
    assert bursts["false_map_observations_with_truth"] > 0
    assert bursts["map_prior_disagreement_rejections"] > 0
    blackout = campaign["scenarios"]["camera_outage"]
    nominal = campaign["scenarios"]["nominal"]
    assert blackout["camera_unavailable_frames"] > nominal["camera_unavailable_frames"]


def test_fixed_seed_reproducibility_and_changed_seed_divergence(campaign):
    repeated = run_campaign(CampaignConfig(seed=14821, frames_per_flight=60))
    assert json.dumps(campaign, sort_keys=True) == json.dumps(repeated, sort_keys=True)
    different = run_campaign(CampaignConfig(seed=14822, frames_per_flight=60))
    assert (
        different["scenarios"]["nominal"]["prior"]["rmse_m"] !=
        campaign["scenarios"]["nominal"]["prior"]["rmse_m"]
    )


def test_campaign_refuses_invalid_experiment_design():
    with pytest.raises(ValueError, match="не хватает|Не хватает"):
        run_campaign(CampaignConfig(fit_flights=2))
    with pytest.raises(ValueError, match="seed"):
        run_campaign(CampaignConfig(seed=-1))
    with pytest.raises(ValueError, match="шаг"):
        run_campaign(CampaignConfig(dt_s=0.0))
