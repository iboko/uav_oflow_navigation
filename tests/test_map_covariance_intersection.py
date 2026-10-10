"""Mathematical regression of 2D covariance intersection (unknown correlation)."""
import numpy as np
import pytest

from src.map_covariance_intersection import (
    covariance_intersection_2d, propose_map_ci_correction
)
from src.orthophoto_localization import MapLocalizationFix


def candidate(n=10.,e=-3.):
    return MapLocalizationFix(True,"GEOMETRICALLY_ACCEPTED_UNVALIDATED",
                              "tile",n,e)


def test_fully_correlated_identical_inputs_cannot_artificially_reduce_variance():
    x,p,w=covariance_intersection_2d(
        (10.,-3.), ((1.,.2),(.2,2.)),
        (10.,-3.), ((1.,.2),(.2,2.))
    )
    assert np.allclose(x, [10.,-3.])
    assert np.allclose(p, [[1.,.2],[.2,2.]],atol=1e-12)
    assert 0 <= w <= 1


def test_complementary_position_uncertainties_can_be_combined_conservatively():
    x,p,w=covariance_intersection_2d(
        (1.,0.), ((.1,0.),(0.,100.)),
        (0.,2.), ((100.,0.),(0.,.1))
    )
    assert 0.2 < w < .8
    assert np.linalg.eigvalsh(p).min() > 0
    assert p[0,0] > .1 and p[1,1] > .1
    assert np.all(np.isfinite(x))


def test_one_far_better_covariance_does_not_fake_extra_confidence():
    x,p,w=covariance_intersection_2d(
        (1.,1.), ((.2,0.),(0.,.2)),
        (100.,100.), ((100.,0.),(0.,100.))
    )
    assert w == 1.
    assert np.allclose(x,(1.,1.))
    assert np.allclose(p, np.eye(2)*.2)


def test_unknown_or_invalid_covariance_and_frame_prevent_proposal():
    args=dict(
        prior_ne_m=(9.9,-3.),
        prior_covariance_ne_m2=((.2,0.),(0.,.2)),
        map_covariance_ne_m2=((.4,0.),(0.,.4)),
        prior_covariance_independently_validated=True,
        map_covariance_independently_validated=True,
        reference_frames_and_timestamps_aligned=True,
        map_integrity_checked=True
    )
    result=propose_map_ci_correction(candidate(), **args)
    assert result.eligible
    assert result.status=="OFFLINE_CI_CANDIDATE_NOT_APPLIED"
    assert result.applied_to_eskf is False
    for flag, status in (
        ("map_integrity_checked", "MAP_INTEGRITY_NOT_ESTABLISHED"),
        ("reference_frames_and_timestamps_aligned", "MAP_FRAME_OR_TIME_UNALIGNED"),
        ("prior_covariance_independently_validated", "PRIOR_COVARIANCE_UNVALIDATED"),
        ("map_covariance_independently_validated", "MAP_COVARIANCE_UNVALIDATED"),
    ):
        k=dict(args);k[flag]=False
        test=propose_map_ci_correction(candidate(),**k)
        assert not test.eligible and test.status == status


def test_covariance_intersection_remains_non_actuating_with_invalid_prior():
    result=propose_map_ci_correction(
        candidate(),
        prior_ne_m=(9.9,-3.),
        prior_covariance_ne_m2=((0.,0.),(0.,0.)),
        map_covariance_ne_m2=((.1,0.),(0.,.1)),
        prior_covariance_independently_validated=True,
        map_covariance_independently_validated=True,
        reference_frames_and_timestamps_aligned=True,
        map_integrity_checked=True
    )
    assert not result.eligible
    assert result.status=="INVALID_POSITION_OR_COVARIANCE"
    assert not result.applied_to_eskf


def test_very_large_disagreement_does_not_get_hidden_by_ci():
    result=propose_map_ci_correction(
        candidate(100.,100.),
        prior_ne_m=(0.,0.),
        prior_covariance_ne_m2=((10.,0.),(0.,10.)),
        map_covariance_ne_m2=((1.,0.),(0.,1.)),
        prior_covariance_independently_validated=True,
        map_covariance_independently_validated=True,
        reference_frames_and_timestamps_aligned=True,
        map_integrity_checked=True,
        max_position_disagreement_m=5.
    )
    assert result.status=="MAP_POSITION_DISAGREEMENT"


def test_singular_and_nonsymmetric_noise_rejected():
    for cov in (
        ((0.,0.),(0.,0.)),
        ((1.,.4),(.2,1.)),
        ((float("nan"),0.),(0.,1.))
    ):
        with pytest.raises(ValueError):
            covariance_intersection_2d(
                (0.,0.), ((1.,0.),(0.,1.)),
                (0.,0.), cov
            )
