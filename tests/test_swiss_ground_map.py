"""Ground-only map: nDSM on the map grid, dilated elevated mask, blurred fill."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import swiss_ground_map as sg  # noqa: E402
from swiss_scene import WORLD_LAT, WORLD_LON, wgs84_to_lv95  # noqa: E402


def test_object_heights_land_on_the_right_map_pixels():
    e0, n0 = wgs84_to_lv95(WORLD_LAT, WORLD_LON)
    dtm = np.full((2000, 2000), 450.0, np.float32)
    dsm = dtm.copy()
    # A 10 m high, 10 x 10 m block whose south-west corner is 20 m north and
    # 30 m east of the world origin.
    c0 = int(round((e0 + 30.0 - sg.DSM_TILE_EAST_M) / 0.5))
    r0 = int(round((sg.DSM_TILE_NORTH_TOP_M - (n0 + 30.0)) / 0.5))
    dsm[r0:r0 + 20, c0:c0 + 20] += 10.0
    georef = {"origin_ne_m": [100.0, -100.0], "matrix_ne_m_per_px": [[0.0, -0.25], [0.25, 0.0]]}
    h = sg.object_heights_on_map(dsm, dtm, georef, (800, 800))
    # Map pixel of local (N 25, E 35): row (100 - 25) / 0.25, col (35 + 100) / 0.25.
    assert h[300, 540] > 9.0
    assert h[300, 480] < 0.5 and h[200, 540] < 0.5


def test_mask_is_dilated_in_proportion_to_height_and_fill_is_smooth():
    h = np.zeros((200, 200), np.float32)
    h[90:110, 90:110] = 20.0
    # Dilation uses the lower bound of the height class: 20 m tall is in the
    # (10, 20] class -> 0.25 * 10 m = 2.5 m = 10 px.
    m = sg.elevated_mask(h, 0.25, h_min=2.0, k=0.25)
    assert m[100, 100] and m[100, 81] and not m[100, 78]
    gray = (np.indices((200, 200)).sum(0) % 2 * 255).astype(np.uint8)  # checkerboard: all edges
    g = sg.ground_map(gray, m, 0.25)
    assert g[m].std() < 0.2 * gray.std() and np.array_equal(g[~m], gray[~m])
