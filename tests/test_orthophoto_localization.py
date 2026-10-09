"""Synthetic georeferenced mosaic tests, not proof of 100m flight accuracy."""
import cv2
import numpy as np
import pytest

from src.ground_visual_motion import CameraCalibration
from src.camera_rotation_compensation import CameraMountCalibration, body_to_ned
from src.orthophoto_localization import (
    MapGeoReference, MapLocalizationFix, OrthophotoTile,
    OrthophotoLocalizer, _ground_ray_offset,
)
from src.map_correction_gate import assess_map_correction


def camera():
    return CameraCalibration(
        1000., 1000., 320., 240., ((0., -1.), (1., 0.))
    )


def mount():
    return CameraMountCalibration(
        ((0., -1., 0.), (1., 0., 0.), (0., 0., 1.))
    )


def map_texture():
    rng = np.random.default_rng(5821)
    image = np.zeros((1100, 1280), np.uint8)
    for _ in range(2900):
        x = int(rng.integers(12, 1268))
        y = int(rng.integers(12, 1088))
        intensity = int(rng.integers(95, 255))
        if rng.random() < .5:
            cv2.circle(image, (x, y), int(rng.integers(2, 5)), intensity, -1)
        else:
            cv2.rectangle(image, (x-2, y-2), (x+3, y+3), intensity, 1)
    return image


def reference():
    return MapGeoReference((100., -50.), ((0., -.1), (.1, 0.)))


def build(tile_image, *, origin=(100., -50.)):
    tile = OrthophotoTile(
        "tile_A", tile_image,
        MapGeoReference(origin, ((0., -.1), (.1, 0.)))
    )
    return OrthophotoLocalizer([tile], camera(), mount())


def frame(texture):
    return texture[260:740, 240:880].copy()


def test_exact_georeferenced_crop_recovers_optical_center_position_at_100m():
    tex = map_texture()
    loc = build(tex)
    fix = loc.localize(
        frame(tex), height_agl_m=100.,
        roll_rad=0., pitch_rad=0., yaw_rad=0.,
    )
    assert fix.accepted, fix
    assert fix.status == "GEOMETRICALLY_ACCEPTED_UNVALIDATED"
    assert fix.tile_name == "tile_A"
    assert fix.north_m == pytest.approx(50., abs=0.40)
    assert fix.east_m == pytest.approx(6., abs=0.40)
    assert fix.inlier_points >= 25
    assert fix.inlier_fraction >= .45
    assert fix.geometric_jacobian_error <= .30
    assert fix.covariance_validated is False


def test_unknown_repeated_tile_causes_safe_ambiguity_rejection():
    texture = map_texture()
    first = OrthophotoTile("A", texture, reference())
    other = OrthophotoTile(
        "B", texture.copy(),
        MapGeoReference((500., 900.), ((0., -.1), (.1, 0.)))
    )
    loc = OrthophotoLocalizer([first, other], camera(), mount())
    fix = loc.localize(
        frame(texture), height_agl_m=100., roll_rad=0.,
        pitch_rad=0., yaw_rad=0.,
    )
    assert not fix.accepted
    assert fix.status == "AMBIGUOUS_MAP_MATCH"
    assert np.isnan(fix.north_m)


def test_blank_camera_frame_and_excess_tilt_fail_closed():
    texture = map_texture()
    loc = build(texture)
    no_texture = loc.localize(
        np.full((480,640), 128, np.uint8), height_agl_m=100.,
        roll_rad=0., pitch_rad=0., yaw_rad=0.,
    )
    assert no_texture.status == "INSUFFICIENT_TEXTURE"
    assert not no_texture.accepted
    tilt = loc.localize(
        frame(texture), height_agl_m=100.,
        roll_rad=0.18, pitch_rad=0., yaw_rad=0.,
    )
    assert tilt.status == "UNSUPPORTED_ATTITUDE"
    assert not tilt.accepted


def test_wrong_map_resolution_yaw_and_reflection_rejected_by_geometric_gate():
    texture = map_texture()
    tile = OrthophotoTile(
        "wrong_scale", texture,
        MapGeoReference((100., 0.), ((0., -.3), (.3, 0.)))
    )
    loc = OrthophotoLocalizer([tile], camera(), mount())
    wrong = loc.localize(frame(texture), height_agl_m=100.,
                         roll_rad=0., pitch_rad=0., yaw_rad=0.)
    assert wrong.status == "NO_GEOMETRICALLY_VALID_MAP_MATCH"
    # Same crop with wrong yaw is not accepted as metrically coherent pose.
    loc = build(texture)
    yaw = loc.localize(frame(texture), height_agl_m=100.,
                       roll_rad=0., pitch_rad=0., yaw_rad=.5)
    assert not yaw.accepted
    wrong_orientation = OrthophotoTile(
        "mirrored", texture,
        MapGeoReference((100., 0.), ((0., .1), (.1, 0.)))
    )
    reflected = OrthophotoLocalizer([wrong_orientation], camera(), mount()).localize(
        frame(texture), height_agl_m=100., roll_rad=0.,
        pitch_rad=0., yaw_rad=0.
    )
    assert not reflected.accepted


def test_projected_ground_footprint_and_camera_lever_arm_are_distinct():
    texture = map_texture()
    lever=(0.35, -0.15, 0.08)
    loc = OrthophotoLocalizer(
        [OrthophotoTile("A", texture, reference())], camera(), mount(),
        camera_offset_body_m=lever,
    )
    roll, pitch, yaw = .03, -.04, 0.
    result=loc.localize(frame(texture), height_agl_m=100.,
                        roll_rad=roll, pitch_rad=pitch, yaw_rad=yaw)
    assert result.accepted, result
    rot = body_to_ned(roll, pitch, yaw)
    principal = _ground_ray_offset(320, 240, 100., camera(),
                                   rot @ np.array(mount().body_from_camera))
    lever_nav = (rot @ np.array(lever))[:2]
    assert result.north_m == pytest.approx(50.-principal[0]-lever_nav[0], abs=.45)
    assert result.east_m == pytest.approx(6.-principal[1]-lever_nav[1], abs=.45)


def test_georeference_must_be_local_metre_frame_not_degrees_or_singular():
    geo=MapGeoReference((0.,0.),((1e-9,0.),(0.,1e-9)))
    with pytest.raises(ValueError,match="разрешение"):
        geo.jacobian()
    geo=MapGeoReference((0.,0.),((0.,0.),(0.,0.)))
    with pytest.raises(ValueError):
        geo.transform(np.array([1.,1.]))


def test_map_fix_alone_is_not_fused_without_measured_covariance():
    tex=map_texture()
    fix=build(tex).localize(frame(tex),height_agl_m=100.,
                              roll_rad=0.,pitch_rad=0.,yaw_rad=0.)
    assert fix.accepted
    proposal=assess_map_correction(
        fix,predicted_ne_m=(50.,6.),
        predicted_covariance_ne_m2=((1.,0.),(0.,1.)),
        map_covariance_ne_m2=None,
        map_error_statistics_validated=False,
        measurement_errors_independent_from_prior=True,
    )
    assert not proposal.eligible
    assert proposal.status=="MAP_COVARIANCE_UNVALIDATED"


def test_correlation_between_vo_and_map_image_prevents_naive_ekf_fusion():
    fix=MapLocalizationFix(True,"GEOMETRICALLY_ACCEPTED_UNVALIDATED",
                            "map",50.,6.)
    denied=assess_map_correction(
        fix,predicted_ne_m=(51.,6.),
        predicted_covariance_ne_m2=((1.,0.),(0.,1.)),
        map_covariance_ne_m2=((.2,0.),(0.,.2)),
        map_error_statistics_validated=True,
        measurement_errors_independent_from_prior=False,
    )
    assert denied.status=="VISUAL_CROSS_CORRELATION_UNMODELED"
    assert not denied.eligible


def test_covariance_gate_checks_consistency_without_applying_position_update():
    fix=MapLocalizationFix(True,"GEOMETRICALLY_ACCEPTED_UNVALIDATED",
                           "map",50.,6.)
    args={
        "predicted_covariance_ne_m2":((.04,0.),(0.,.04)),
        "map_covariance_ne_m2":((.04,0.),(0.,.04)),
        "map_error_statistics_validated":True,
        "measurement_errors_independent_from_prior":True,
    }
    good=assess_map_correction(fix,predicted_ne_m=(49.9,6.),**args)
    assert good.eligible
    assert good.status=="CANDIDATE_ONLY_NOT_APPLIED"
    assert good.nis_2d < 9.21
    bad=assess_map_correction(fix,predicted_ne_m=(48.,6.),**args)
    assert not bad.eligible
    assert bad.status=="MAP_INNOVATION_REJECTED"
    assert bad.nis_2d > 9.21
    singular=assess_map_correction(
        fix,predicted_ne_m=(49.,6.),
        predicted_covariance_ne_m2=((0.,0.),(0.,0.)),
        map_covariance_ne_m2=((.04,0.),(0.,.04)),
        map_error_statistics_validated=True,
        measurement_errors_independent_from_prior=True,
    )
    assert singular.status=="INVALID_COVARIANCE"
