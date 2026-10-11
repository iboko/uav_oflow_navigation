"""Bag-slice importer: chunk resynchronisation inside an arbitrary byte range."""
import importlib.util
import struct
from pathlib import Path

import pytest

pytest.importorskip("cv2")
_SPEC = importlib.util.spec_from_file_location(
    "munfrl_bag_slice", Path(__file__).resolve().parents[1] / "tools" / "munfrl_bag_slice.py")
bag = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bag)


def _field(key: str, value: bytes) -> bytes:
    body = key.encode() + b"=" + value
    return struct.pack("<I", len(body)) + body


def _record(fields: bytes, data: bytes) -> bytes:
    return struct.pack("<I", len(fields)) + fields + struct.pack("<I", len(data)) + data


def _message(conn: int, sec: int, payload: bytes) -> bytes:
    return _record(_field("op", b"\x02") + _field("conn", struct.pack("<I", conn)) +
                   _field("time", struct.pack("<II", sec, 0)), payload)


def _chunk(inner: bytes) -> bytes:
    return _record(_field("compression", b"none") + _field("op", b"\x05") +
                   _field("size", struct.pack("<I", len(inner))), inner)


def test_chunks_resync_from_the_middle_of_a_bag_and_skip_a_truncated_tail():
    msgs = [_message(3, 100 + k, bytes([k]) * 7) for k in range(3)]
    first, second = _chunk(msgs[0] + msgs[1]), _chunk(msgs[2])
    # A slice starts inside an unrelated record and ends inside a chunk.
    data = b"\x01garbage-op=\x05" + first + second + second[:-3]
    found = [body for _, body in bag.chunks(data)]
    assert found == [msgs[0] + msgs[1], msgs[2]]
    records = list(bag._records(found[0], 0, len(found[0])))
    assert [h["op"] for h, _ in records] == [b"\x02", b"\x02"]
    assert records[1][1] == bytes([1]) * 7


def test_gga_parsing_and_hemispheres():
    t, lat, lon, quality, alt = bag._gga(
        "$GNGGA,191116.20,4728.82000,N,05300.56771,W,4,12,0.66,32.5,M,7.8,M,1.2,0000*7A")
    assert t == pytest.approx(19 * 3600 + 11 * 60 + 16.2)
    assert lat == pytest.approx(47 + 28.82 / 60)
    assert lon == pytest.approx(-(53 + 0.56771 / 60))
    assert (quality, alt) == (4, 32.5)
    assert bag._gga("$GNRMC,191117.00,A,4728.82000,N,05300.56771,W") is None


def test_quaternion_from_matrix_is_normalised_and_round_trips():
    import numpy as np
    c, s = np.cos(2.5), np.sin(2.5)
    r = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]) @ np.diag([1., -1., -1.])
    w, x, y, z = bag._quat_from_matrix(r)
    assert w >= 0 and w * w + x * x + y * y + z * z == pytest.approx(1.)
    back = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    assert np.allclose(back, r, atol=1e-9)
