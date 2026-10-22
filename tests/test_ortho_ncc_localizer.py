"""NCC по ортотрансформированному кадру: другой «сезон», повторяющаяся текстура."""
import math

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from src.camera_rotation_compensation import CameraMountCalibration, body_to_ned  # noqa: E402
from src.ground_visual_motion import CameraCalibration  # noqa: E402
from src.ortho_ncc_localizer import OrthoNccLocalizer  # noqa: E402
from src.orthophoto_localization import MapGeoReference  # noqa: E402

GSD = 0.25
CAL = CameraCalibration(233.9, 233.9, 160.0, 120.0, ((0.0, -1.0), (1.0, 0.0)))
MOUNT = CameraMountCalibration(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
GEO = MapGeoReference((100.0 - GSD / 2, -100.0 + GSD / 2), ((0.0, -GSD), (GSD, 0.0)))  # 200 m, centred


def _texture(seed=2, n=800):
    rng = np.random.default_rng(seed)
    img = np.zeros((n, n), np.float32)
    for cells, w in ((10, 0.5), (40, 1.0), (160, 0.7), (400, 0.4)):
        img += w * cv2.resize(rng.random((cells, cells)).astype(np.float32), (n, n), interpolation=cv2.INTER_CUBIC)
    return (255 * (img - img.min()) / (img.max() - img.min())).astype(np.uint8)


def _render(world, n, e, h, roll, pitch, yaw):
    """Camera frame 320x240 of the ground texture (pinhole, flat ground)."""
    r_nc = body_to_ned(roll, pitch, yaw) @ np.asarray(MOUNT.body_from_camera)
    k_inv = np.linalg.inv(np.array([[CAL.focal_x_px, 0, CAL.center_x_px], [0, CAL.focal_y_px, CAL.center_y_px],
                                    [0, 0, 1.0]]))
    us, vs = np.meshgrid(np.arange(320), np.arange(240))
    rays = np.stack([us, vs, np.ones_like(us)], -1).astype(float) @ k_inv.T @ r_nc.T
    s = h / rays[..., 2]
    gn, ge = n + s * rays[..., 0], e + s * rays[..., 1]
    col = ((ge - GEO.origin_ne_m[1]) / GSD).astype(np.float32)
    row = ((GEO.origin_ne_m[0] - gn) / GSD).astype(np.float32)
    return cv2.remap(world, col, row, cv2.INTER_LINEAR)


def test_ncc_localizes_a_frame_against_a_different_season_map():
    world = _texture()
    # "Other year": gamma, contrast, blur and noise on the map.
    rng = np.random.default_rng(5)
    old = np.clip(255 * (cv2.GaussianBlur(world, (0, 0), 1.2) / 255.0) ** 1.6 * 0.8 + 20
                  + rng.normal(0, 6, world.shape), 0, 255).astype(np.uint8)
    loc = OrthoNccLocalizer(old, GEO, CAL, MOUNT)
    n, e, h, r, p, y = 12.0, -8.0, 40.0, 0.03, -0.02, math.radians(80)
    frame = _render(world, n, e, h, r, p, y)
    fix = loc.localize(frame, prior_ne_m=(n + 9.0, e - 6.0), radius_m=20.0, height_agl_m=h,
                       roll_rad=r, pitch_rad=p, yaw_rad=y)
    assert fix.accepted, (fix, loc.last_peak, loc.last_margin)
    assert math.hypot(fix.north_m - n, fix.east_m - e) < 0.3


def test_ncc_refuses_repetitive_texture_and_missing_prior():
    tile = _texture(n=80)
    world = np.tile(tile, (10, 10))  # 20 m period: several equal peaks in the window
    loc = OrthoNccLocalizer(world, GEO, CAL, MOUNT)
    frame = _render(world, 0.0, 0.0, 40.0, 0.0, 0.0, 0.0)
    fix = loc.localize(frame, prior_ne_m=(0.0, 0.0), radius_m=25.0, height_agl_m=40.0,
                       roll_rad=0.0, pitch_rad=0.0, yaw_rad=0.0)
    assert not fix.accepted and fix.status == "AMBIGUOUS_MAP_MATCH"
    assert loc.localize(frame, prior_ne_m=None, radius_m=25.0, height_agl_m=40.0, roll_rad=0.0, pitch_rad=0.0,
                        yaw_rad=0.0).status == "NO_PRIOR_POSITION"


def test_ncc_uses_camera_height_and_full_rotated_lever_arm():
    world = _texture(seed=7)
    offset = np.array([1.5, -1.0, 2.0])
    n, e, h, r, p, y = 12.0, -8.0, 40.0, 0.04, -0.03, math.radians(65)
    lever = body_to_ned(r, p, y) @ offset
    frame = _render(world, n + lever[0], e + lever[1], h - lever[2], r, p, y)
    loc = OrthoNccLocalizer(world, GEO, CAL, MOUNT, camera_offset_body_m=offset, patch_m=20.0,
                            yaw_search_deg=(0.0,))
    fix = loc.localize(frame, prior_ne_m=(n + 3.0, e - 2.0), radius_m=10.0,
                       height_agl_m=h, roll_rad=r, pitch_rad=p, yaw_rad=y)
    assert fix.accepted, fix
    assert math.hypot(fix.north_m - n, fix.east_m - e) < 0.3


def test_ambiguity_at_another_heading_is_rejected(monkeypatch):
    world = _texture()
    loc = OrthoNccLocalizer(world, GEO, CAL, MOUNT, yaw_search_deg=(0.0, 2.0))
    frame = _render(world, 0, 0, 40, 0, 0, 0)
    surfaces = []
    # Each heading is individually unambiguous, but distinct positions at
    # the two headings have almost equal scores.
    def match(window, patch, method):
        response = np.zeros((window.shape[0] - patch.shape[0] + 1,
                             window.shape[1] - patch.shape[1] + 1), np.float32)
        response[15, 15 if not surfaces else 40] = 0.9 if not surfaces else 0.88
        surfaces.append(response)
        return response
    monkeypatch.setattr(cv2, "matchTemplate", match)
    fix = loc.localize(frame, prior_ne_m=(0.0, 0.0), radius_m=20.0,
                       height_agl_m=40, roll_rad=0, pitch_rad=0, yaw_rad=0)
    assert not fix.accepted and fix.status == "AMBIGUOUS_MAP_MATCH"


@pytest.mark.parametrize("kwargs", [{"yaw_search_deg": ()}, {"patch_m": 0},
                                    {"camera_offset_body_m": (0, 0, float("nan"))}])
def test_invalid_ncc_geometry_is_refused(kwargs):
    with pytest.raises(ValueError):
        OrthoNccLocalizer(_texture(), GEO, CAL, MOUNT, **kwargs)


@pytest.mark.parametrize("heading_error_deg", [-7.0, 0.0, 5.0])
def test_ncc_measures_heading_of_the_best_rotation(heading_error_deg):
    world = _texture()
    loc = OrthoNccLocalizer(world, GEO, CAL, MOUNT)
    n, e, h, y = 5.0, 3.0, 40.0, math.radians(80)
    frame = _render(world, n, e, h, 0.0, 0.0, y)
    fix = loc.localize(frame, prior_ne_m=(n, e), radius_m=10.0, height_agl_m=h,
                       roll_rad=0.0, pitch_rad=0.0, yaw_rad=y + math.radians(heading_error_deg))
    assert fix.accepted
    err = math.degrees(math.atan2(math.sin(fix.yaw_rad - y), math.cos(fix.yaw_rad - y)))
    assert abs(err) < 1.0, err


def test_ncc_heading_is_nan_at_the_edge_of_the_search():
    world = _texture()
    loc = OrthoNccLocalizer(world, GEO, CAL, MOUNT, yaw_search_deg=(-2.0, 0.0, 2.0), yaw_extension_deg=0.0)
    y = math.radians(80)
    frame = _render(world, 0.0, 0.0, 40.0, 0.0, 0.0, y)
    fix = loc.localize(frame, prior_ne_m=(0.0, 0.0), radius_m=10.0, height_agl_m=40.0,
                       roll_rad=0.0, pitch_rad=0.0, yaw_rad=y + math.radians(6.0))
    assert not math.isfinite(fix.yaw_rad) or not fix.accepted


def test_ncc_search_extends_past_the_edge_and_recovers_the_heading():
    world = _texture()
    loc = OrthoNccLocalizer(world, GEO, CAL, MOUNT, yaw_search_deg=(-2.0, 0.0, 2.0), yaw_extension_deg=10.0)
    y = math.radians(80)
    frame = _render(world, 0.0, 0.0, 40.0, 0.0, 0.0, y)
    fix = loc.localize(frame, prior_ne_m=(0.0, 0.0), radius_m=10.0, height_agl_m=40.0,
                       roll_rad=0.0, pitch_rad=0.0, yaw_rad=y + math.radians(6.0))
    assert fix.accepted and math.isfinite(fix.yaw_rad)
    assert abs(math.degrees(fix.yaw_rad - y)) < 1.0
    # The extension stops once the peak is passed: not all 5 extra rotations.
    assert max(abs(math.degrees(k)) for k in loc.last_peaks) <= 10.0 + 1e-6


def _render_surface(world, surface, n, e, h_cam_up, roll, pitch, yaw, steps=400):
    """Ray-marched camera frame of a textured height field (surface: map grid, up)."""
    r_nc = body_to_ned(roll, pitch, yaw) @ np.asarray(MOUNT.body_from_camera)
    k_inv = np.linalg.inv(np.array([[CAL.focal_x_px, 0, CAL.center_x_px], [0, CAL.focal_y_px, CAL.center_y_px],
                                    [0, 0, 1.0]]))
    us, vs = np.meshgrid(np.arange(320), np.arange(240))
    rays = np.stack([us, vs, np.ones_like(us)], -1).astype(float) @ k_inv.T @ r_nc.T
    rays /= rays[..., 2:3]                           # unit down component
    hit = np.full(us.shape, np.nan)
    for s in np.linspace(0.0, h_cam_up + 5.0, steps):
        gn, ge = n + s * rays[..., 0], e + s * rays[..., 1]
        col = ((ge - GEO.origin_ne_m[1]) / GSD).astype(np.float32)
        row = ((GEO.origin_ne_m[0] - gn) / GSD).astype(np.float32)
        z = cv2.remap(surface, col, row, cv2.INTER_LINEAR)
        new = np.isnan(hit) & (h_cam_up - s <= z)
        hit[new] = s
    gn, ge = n + hit * rays[..., 0], e + hit * rays[..., 1]
    col = ((ge - GEO.origin_ne_m[1]) / GSD).astype(np.float32)
    row = ((GEO.origin_ne_m[0] - gn) / GSD).astype(np.float32)
    return cv2.remap(world, col, row, cv2.INTER_LINEAR)


def test_dsm_projection_removes_building_parallax():
    world = _texture(seed=3)
    surface = np.zeros(world.shape, np.float32)
    # A 15 m block 8-18 m east of the camera, 12 m wide: strong parallax at 40 m.
    r0, c0 = int((100.0 - 6.0) / GSD), int((8.0 + 100.0) / GSD)
    surface[r0:r0 + 48, c0:c0 + 40] = 15.0
    # Roof texture distinct from the ground below it (as on a real map).
    world[r0:r0 + 48, c0:c0 + 40] = np.clip(world[r0:r0 + 48, c0:c0 + 40].astype(int) + 60, 0, 255).astype(np.uint8)
    n, e, h, y = 0.0, 0.0, 40.0, math.radians(80)
    frame = _render_surface(world, surface, n, e, h, 0.0, 0.0, y)
    flat = OrthoNccLocalizer(world, GEO, CAL, MOUNT, yaw_search_deg=(0.0,))
    dsm = OrthoNccLocalizer(world, GEO, CAL, MOUNT, yaw_search_deg=(0.0,), surface_m=surface)
    kw = dict(prior_ne_m=(n, e), radius_m=8.0, height_agl_m=h, roll_rad=0.0, pitch_rad=0.0, yaw_rad=y)
    f_flat, f_dsm = flat.localize(frame, **kw), dsm.localize(frame, **kw)
    assert f_dsm.accepted
    err_dsm = math.hypot(f_dsm.north_m - n, f_dsm.east_m - e)
    err_flat = math.hypot(f_flat.north_m - n, f_flat.east_m - e) if f_flat.accepted else float("inf")
    assert err_dsm < 0.25 and dsm.last_peak > flat.last_peak, (err_dsm, err_flat, dsm.last_peak, flat.last_peak)


def test_patch_at_boresight_keeps_a_tilted_frame_usable():
    # Pitch 0.19 rad at 50 m: the image edge on the near side is 14.6 m
    # from the nadir, inside the 15 m half patch; centred on the optical
    # axis the patch fits and the fix is still the camera position.
    world = _texture(seed=4)
    n, e, h, r, p, y = 5.0, 3.0, 50.0, 0.0, 0.19, math.radians(30)
    frame = _render(world, n, e, h, r, p, y)
    kw = dict(prior_ne_m=(n + 2.0, e - 2.0), radius_m=8.0, height_agl_m=h, roll_rad=r, pitch_rad=p, yaw_rad=y)
    nadir = OrthoNccLocalizer(world, GEO, CAL, MOUNT, max_tilt_rad=0.2)
    assert nadir.localize(frame, **kw).status == "NO_USABLE_ORTHO_PATCH"
    bore = OrthoNccLocalizer(world, GEO, CAL, MOUNT, max_tilt_rad=0.2, patch_at_boresight=True)
    fix = bore.localize(frame, **kw)
    assert fix.accepted, (fix, bore.last_peak, bore.last_margin)
    assert math.hypot(fix.north_m - n, fix.east_m - e) < 0.3
