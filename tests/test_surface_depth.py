import numpy as np
import pytest

pytest.importorskip("cv2")

from src.surface_depth import SurfaceDepth

NADIR = ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))   # camera of the SITL companion


def depths(surface, pts, rpy=(0.0, 0.0, 0.0), alt=50.0, ne=(0.0, 0.0)):
    return surface.feature_depths(np.asarray(pts, float), focal_px=(200.0, 200.0), center_px=(160.0, 120.0),
                                  body_from_camera=np.asarray(NADIR), rpy=rpy, position_ne_m=ne, altitude_m=alt)


def flat(height=0.0):
    return SurfaceDepth(np.full((400, 400), height, np.float32), (100.0, -100.0), 0.5)


def test_flat_ground_nadir_and_tilted():
    s = flat(2.0)
    assert depths(s, [[160, 120], [10, 10]]) == pytest.approx([48.0, 48.0], abs=1e-3)
    # Pitched 0.1 rad: the optical-axis ray is longer by 1/cos.
    d = depths(s, [[160, 120]], rpy=(0.0, 0.1, 0.0))
    assert d[0] == pytest.approx(48.0 / np.cos(0.1), rel=1e-3)


def test_roof_under_part_of_the_frame():
    grid = np.zeros((400, 400), np.float32)
    grid[:, 200:] = 20.0                     # east half (from the origin): a 20 m roof
    s = SurfaceDepth(grid, (100.0, -100.0), 0.5)
    # Image right = body right = east at zero yaw.
    d = depths(s, [[60, 120], [300, 120]], ne=(0.0, 0.0))
    assert d[0] == pytest.approx(50.0, abs=0.05)
    assert d[1] == pytest.approx(30.0, abs=0.05)


def test_georef_and_upward_ray():
    s = SurfaceDepth.from_georef(np.zeros((10, 10), np.float32),
                                 {"origin_ne_m": [5.0, -5.0], "matrix_ne_m_per_px": [[0.0, -1.0], [1.0, 0.0]]})
    assert s.height_at(0.0, 0.0) == pytest.approx([0.0])
    with pytest.raises(ValueError):
        depths(s, [[160, 120]], rpy=(0.0, 1.6, 0.0))
    with pytest.raises(ValueError):
        SurfaceDepth.from_georef(np.zeros((4, 4)), {"origin_ne_m": [0, 0], "matrix_ne_m_per_px": [[1, 0], [0, 1]]})
