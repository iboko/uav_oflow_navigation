"""Depth of image features from an onboard surface model (DSM).

The planar VO scales image motion by the rangefinder height: the distance
to the one surface point below the vehicle. Over relief and roofs the
tracked features lie at other depths (3D swiss scene: the predicted scale
error correlates with the VO velocity error, r ~ 0.5). Here each feature
ray is cast onto the DSM from the camera position and altitude, giving
its depth along the optical axis.

Inputs that come from the estimator (horizontal position, altitude above
the DSM datum, attitude) carry their own errors; a position error of a few
metres moves the lookup but not the camera altitude.
"""
from __future__ import annotations

import cv2
import numpy as np

from .camera_rotation_compensation import body_to_ned


class SurfaceDepth:
    """DSM heights (m, up, same datum as the altitude) on a north-up grid.

    ``origin_ne_m`` is the (north, east) of pixel (row 0, col 0) and
    ``gsd_m`` the cell size; rows grow south, columns grow east.
    """

    def __init__(self, surface_m: np.ndarray, origin_ne_m: tuple[float, float], gsd_m: float,
                 iterations: int = 8) -> None:
        s = np.asarray(surface_m, dtype=np.float32)
        if s.ndim != 2 or not np.isfinite(s).all() or gsd_m <= 0:
            raise ValueError("surface must be a finite 2D grid with a positive cell size")
        self.surface = s
        self.origin = (float(origin_ne_m[0]), float(origin_ne_m[1]))
        self.gsd = float(gsd_m)
        self.iterations = iterations

    @classmethod
    def from_georef(cls, surface_m: np.ndarray, georef: dict) -> "SurfaceDepth":
        m = np.asarray(georef["matrix_ne_m_per_px"], dtype=float)
        if not np.allclose(m, [[0.0, -m[1, 0]], [m[1, 0], 0.0]]) or m[1, 0] <= 0:
            raise ValueError("only north-up grids are supported")
        return cls(surface_m, tuple(georef["origin_ne_m"]), float(m[1, 0]))

    def height_at(self, north, east) -> np.ndarray:
        n = np.atleast_1d(np.asarray(north, dtype=np.float32))
        e = np.atleast_1d(np.asarray(east, dtype=np.float32))
        col = ((e - self.origin[1]) / self.gsd).reshape(1, -1)
        row = ((self.origin[0] - n) / self.gsd).reshape(1, -1)
        return cv2.remap(self.surface, col.astype(np.float32), row.astype(np.float32), cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE).reshape(n.shape)

    def feature_depths(self, points_px: np.ndarray, *, focal_px: tuple[float, float],
                       center_px: tuple[float, float], body_from_camera: np.ndarray,
                       rpy: tuple[float, float, float], position_ne_m: tuple[float, float],
                       altitude_m: float) -> np.ndarray:
        """Depth (m, along the optical axis) of each pixel's ray on the DSM."""
        p = np.asarray(points_px, dtype=np.float64).reshape(-1, 2)
        rays_c = np.column_stack([(p[:, 0] - center_px[0]) / focal_px[0],
                                  (p[:, 1] - center_px[1]) / focal_px[1], np.ones(len(p))])
        r_nc = body_to_ned(*rpy) @ np.asarray(body_from_camera, dtype=np.float64)
        rays = rays_c @ r_nc.T                     # NED, camera z component 1
        down = rays[:, 2]
        if np.any(down <= 0.05):
            raise ValueError("ray does not point at the ground")
        n0, e0 = float(position_ne_m[0]), float(position_ne_m[1])
        z = np.full(len(p), float(self.height_at(n0, e0)[0]))
        s = (altitude_m - z) / down
        for k in range(self.iterations):
            z = self.height_at(n0 + s * rays[:, 0], e0 + s * rays[:, 1]).astype(np.float64)
            # Plain fixed point first (one step on a flat cell), then damped:
            # at a wall the plain step oscillates between roof and ground.
            w = 1.0 if k < 3 else 0.5
            s = (1.0 - w) * s + w * (altitude_m - z) / down
        return s
