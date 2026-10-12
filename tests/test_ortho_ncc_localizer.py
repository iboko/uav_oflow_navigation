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
