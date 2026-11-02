"""VO at large tilt: optical-axis depth and level-motion conversion to NED."""
import math

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from src.camera_rotation_compensation import (
    CameraMountCalibration,
    body_to_ned,
    camera_delta_to_center_ned,
)
from src.continuous_visual_odometry import (
    ContinuousPlanarOdometry,
    VisualFrame,
)
from src.ground_visual_motion import CameraCalibration

CAL = CameraCalibration(233.9, 233.9, 160.0, 120.0, ((0.0, -1.0), (1.0, 0.0)))
MOUNT = CameraMountCalibration(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
GSD = 0.1


def _texture(seed=3, n=1600):
    rng = np.random.default_rng(seed)
    img = np.zeros((n, n), np.float32)
    for cells, w in ((20, 0.5), (80, 1.0), (320, 0.7), (800, 0.4)):
        img += w * cv2.resize(rng.random((cells, cells)).astype(np.float32), (n, n),
                              interpolation=cv2.INTER_CUBIC)
    return (255 * (img - img.min()) / (img.max() - img.min())).astype(np.uint8)


def _render(world, n, e, h, roll, pitch, yaw):
    """320x240 pinhole frame of flat ground ``h`` m below the camera; world centred at 0."""
    r_nc = body_to_ned(roll, pitch, yaw) @ np.asarray(MOUNT.body_from_camera)
    k_inv = np.linalg.inv(np.array([[CAL.focal_x_px, 0, CAL.center_x_px],
                                    [0, CAL.focal_y_px, CAL.center_y_px], [0, 0, 1.0]]))
    us, vs = np.meshgrid(np.arange(320), np.arange(240))
    rays = np.stack([us, vs, np.ones_like(us)], -1).astype(float) @ k_inv.T @ r_nc.T
    s = h / rays[..., 2]
    gn, ge = n + s * rays[..., 0], e + s * rays[..., 1]
    half = world.shape[0] / 2
    return cv2.remap(world, (half + ge / GSD).astype(np.float32),
                     (half - gn / GSD).astype(np.float32), cv2.INTER_LINEAR)


@pytest.mark.parametrize("level", [False, True])
def test_level_motion_inverts_body_projection_of_horizontal_motion(level):
    rpy0, rpy1 = (0.02, -0.31, 0.0), (0.025, -0.305, 0.01)
    lever = (0.1, 0.02, 0.05)
    d_center = np.array([1.3, 0.1, 0.0])
    r0, r1 = body_to_ned(*rpy0), body_to_ned(*rpy1)
    d_cam = d_center + (r1 - r0) @ np.asarray(lever)
    xy = (r0.T @ d_cam)[:2]
    got = camera_delta_to_center_ned(tuple(xy), rpy0, rpy1, lever, level_motion=level)
    if level:
        assert np.allclose(got, d_center[:2], atol=1e-12)
    else:
        # The old zero body-z assumption loses about (1 - cos^2 tilt) of it.
        assert np.linalg.norm(got) < 0.95 * np.linalg.norm(d_center[:2])


def test_level_motion_uses_the_given_vertical_delta():
    rpy = (0.0, 0.3, 0.0)
    d = np.array([1.0, 0.0, -0.5])
    xy = (body_to_ned(*rpy).T @ d)[:2]
    got = camera_delta_to_center_ned(tuple(xy), rpy, rpy, (0, 0, 0), level_motion=True,
                                     vertical_delta_m=-0.5)
    assert np.allclose(got, d[:2], atol=1e-12)


def test_tilt_geometry_needs_mount():
    with pytest.raises(ValueError):
        ContinuousPlanarOdometry(CAL, tilt_geometry=True)


def _speed_error(tilt_geometry, roll, pitch):
    world = _texture()
    vo = ContinuousPlanarOdometry(CAL, camera_mount=MOUNT, max_tilt_rad=0.4,
                                  tilt_geometry=tilt_geometry)
    h_cam, yaw, dt = 50.0, 0.4, 0.1
    v = np.array([6.0, 2.0])  # NED m/s, level flight
    out = []
    for k in range(4):
        n, e = v * dt * k
        frame = VisualFrame(int((1 + k * dt) * 1e6), _render(world, n, e, h_cam, roll, pitch, yaw),
                            height_agl_m=h_cam, yaw_rad=yaw, roll_rad=roll, pitch_rad=pitch)
        res = vo.process(frame)
        if k:
            assert res.accepted, res.status
            out.append((res.vn_m_s, res.ve_m_s))
    est = np.mean(out, axis=0)
    return float(np.dot(est - v, v) / np.dot(v, v))


def test_tilt_geometry_removes_along_track_underestimate_at_large_tilt():
    roll, pitch = 0.12, -0.30
    old = _speed_error(False, roll, pitch)
    new = _speed_error(True, roll, pitch)
    c = math.cos(roll) * math.cos(pitch)
    assert old < -0.08  # about 1 - cos^3(tilt) = -13 %
    assert abs(new) < 0.02, (old, new, 1 - c ** 3)
