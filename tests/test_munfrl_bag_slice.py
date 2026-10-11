"""Bag-slice importer: chunk resynchronisation inside an arbitrary byte range."""
import importlib.util
import struct
from pathlib import Path

import numpy as np
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


def _ppk_file(tmp_path, rows):
    path = tmp_path / "track.pos"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_ppk_interpolation_inside_two_fixed_epochs(tmp_path):
    path = _ppk_file(tmp_path, [
        "% GPST latitude longitude height quality",
        "2022/02/22 19:11:16.000 47.0 -53.0 100.0 1",
        "2022/02/22 19:11:16.200 47.2 -52.8 100.4 1",
    ])
    ppk = bag.read_ppk(path)
    lat, lon, h, q = bag.sample_ppk(ppk, 0.5 * (ppk[0, 0] + ppk[1, 0]))
    assert (lat, lon, h, q) == pytest.approx((47.1, -52.9, 100.2, 1))
    assert bag.sample_ppk(ppk, ppk[0, 0])[-1] == 1
    assert bag.sample_ppk(ppk, ppk[-1, 0])[-1] == 1


def test_ppk_categorical_quality_never_interpolates_a_fix(tmp_path):
    ppk = bag.read_ppk(_ppk_file(tmp_path, [
        "2022/02/22 19:11:16.000 47.0 -53.0 100.0 1",
        "2022/02/22 19:11:16.200 47.2 -52.8 100.4 2",
    ]))
    assert bag.sample_ppk(ppk, (ppk[0, 0] + ppk[1, 0]) / 2)[-1] == 0


def test_ppk_extrapolation_and_unsupported_gaps_are_rejected(tmp_path):
    ppk = bag.read_ppk(_ppk_file(tmp_path, [
        "2022/02/22 19:11:16.000 47.0 -53.0 100.0 1",
        "2022/02/22 19:11:16.200 47.2 -52.8 100.4 1",
    ]))
    with pytest.raises(ValueError, match="вне диапазона"):
        bag.sample_ppk(ppk, ppk[0, 0] - .1)
    with pytest.raises(ValueError, match="вне диапазона"):
        bag.sample_ppk(ppk, ppk[-1, 0] + .1)
    ppk[1, 0] += 10.
    with pytest.raises(ValueError, match="Разрыв"):
        bag.sample_ppk(ppk, ppk[0, 0] + .2)


@pytest.mark.parametrize("rows", [
    ["2022/02/22 19:11:16.200 47.0 -53.0 100.0 1",
     "2022/02/22 19:11:16.000 47.2 -52.8 100.4 1"],
    ["2022/02/22 19:11:16.000 47.0 -53.0 100.0 1",
     "2022/02/22 19:11:16.000 47.2 -52.8 100.4 1"],
])
def test_ppk_out_of_order_or_duplicate_epoch_is_rejected(tmp_path, rows):
    with pytest.raises(ValueError, match="дублируются|возрастанию"):
        bag.read_ppk(_ppk_file(tmp_path, rows))


@pytest.mark.parametrize("bad", ["91.0 -53.0 100.0 1", "47.0 -181.0 100.0 1",
                                  "nan -53.0 100.0 1", "47.0 -53.0 100.0 0"])
def test_ppk_invalid_reference_is_rejected(tmp_path, bad):
    with pytest.raises(ValueError, match="строка PPK"):
        bag.read_ppk(_ppk_file(tmp_path, [
            "2022/02/22 19:11:16.000 " + bad,
            "2022/02/22 19:11:16.200 47.2 -52.8 100.4 1",
        ]))


@pytest.mark.parametrize("flag,value", [("--scale", "0"), ("--scale", "nan"),
                                        ("--scale", "-0.5"), ("--frame-step", "0")])
def test_importer_rejects_invalid_cli_parameters_before_file_access(tmp_path, monkeypatch, flag, value):
    import sys
    monkeypatch.setattr(sys, "argv",
                        ["munfrl_bag_slice", "nonexistent.bag", "nonexistent.pos",
                         str(tmp_path / "output"), "--connections-from", "nonexistent.head",
                         flag, value])
    with pytest.raises(SystemExit) as exc:
        bag.main()
    assert exc.value.code == 2
    assert not (tmp_path / "output").exists()


def test_imu_sample_times_follow_sequence_clock_not_receive_bursts():
    rng = np.random.default_rng(3)
    seq = np.arange(1000, 1400)
    seq = np.delete(seq, [50, 51, 200])  # dropped packets
    true_t = 100.0 + (seq - 1000) * 0.0025
    receive = true_t + 0.004 + rng.uniform(0.0, 0.03, len(seq))  # bursts, >= 4 ms lag
    imu = [(float(t), (0., 0., 0., 1.), (0., 0., 0.), (0., 0., -9.8), int(s))
           for t, s in zip(receive, seq)][::-1]
    out, info = bag.sample_times_from_seq(imu)
    fitted = np.array([r[0] for r in out])
    assert [r[4] for r in out] == sorted(seq.tolist())
    assert info["dropped_samples"] == 3
    assert info["rate_hz"] == pytest.approx(400.0, rel=1e-3)
    # Uniform grid; the lower envelope keeps only the minimum transport lag.
    assert np.allclose(np.diff(fitted) / np.diff(seq), 0.0025, rtol=1e-3)
    assert np.max(np.abs(fitted - (true_t + 0.004))) < 0.002


def test_imu_with_implausible_rate_is_refused():
    imu = [(float(k), (0., 0., 0., 1.), (0., 0., 0.), (0., 0., -9.8), k) for k in range(200)]
    with pytest.raises(SystemExit, match="period"):
        bag.sample_times_from_seq(imu)
