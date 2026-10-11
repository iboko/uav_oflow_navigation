#!/usr/bin/env python3
"""Import an ARBITRARY BYTE RANGE of a MUN-FRL ROS1 bag into the VO inputs.

MUN-FRL (Thalagala et al., CC BY 4.0, mun-frl-vil-dataset.readthedocs.io)
DJI M600 sequences are 20-90 GB bags. A mid-flight slice can be fetched
with an HTTP range request; this tool re-synchronises on bag CHUNK records
inside the slice (no bag index needed) and writes:

  frames/*.png        down camera (/camera/image_mono), Kannala-Brandt
                      undistorted to a pinhole image, scaled by --scale
  manifest.csv        timestamp_us,image_path,height_agl_m,true_n_m,true_e_m,
                      true_vn_m_s,true_ve_m_s  (src.run_camera_odometry format)
  attitude.csv        timestamp_us,qw,qx,qy,qz  body->NED (Xsens /imu/data)
  imu.csv             timestamp_us,gx,gy,gz,ax,ay,az  (body, z down)
  camera.yaml         pinhole intrinsics + body_from_camera + lever arm
  slice_info.json     time mapping, counts, reference statistics

Reference: PPK solution (.pos, RTKLIB, GPST) interpolated to the frame
time; horizontal N/E are relative to the first frame. Bag time -> UTC
mapping uses the receiver NMEA GGA sentences inside the slice (minimum
receive latency). Height above ground = PPK height - takeoff PPK height,
i.e. a FLAT-GROUND assumption (open-pit quarry: not exact; reported).
"""
from __future__ import annotations

import argparse
import bz2
import calendar
import csv
import json
import math
import mmap
import struct
from pathlib import Path

import cv2
import numpy as np

# DJI M600 calibration published on the dataset page (Calibration.html).
FX, FY, CX, CY = 854.383024, 853.285954, 780.324522, 520.690672
KB = (-0.07937700, 0.02228435, -0.03852023, 0.01346873)  # k2..k5 = OpenCV fisheye k1..k4
BODY_T_CAMDOWN = np.array([
    [0.00235643, 0.99997843, -0.00613037, -0.25805624],
    [-0.99960218, 0.00218315, -0.02811962, -0.01138283],
    [-0.02810563, 0.0061942, 0.99958577, 0.09243762],
])
GPST_MINUS_UTC_S = 18.0  # 2017-2026
R_NED_ENU = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]])


def _fields(buf: bytes) -> dict:
    out, i = {}, 0
    while i < len(buf):
        n = struct.unpack_from("<I", buf, i)[0]
        i += 4
        if n == 0 or i + n > len(buf):
            raise ValueError("bad field")
        key, _, value = buf[i:i + n].partition(b"=")
        out[key.decode("ascii")] = value
        i += n
    return out


def _records(data, pos: int, end: int):
    while pos + 8 <= end:
        hl = struct.unpack_from("<I", data, pos)[0]
        if pos + 4 + hl + 4 > end:
            return
        header = _fields(bytes(data[pos + 4:pos + 4 + hl]))
        pos += 4 + hl
        dl = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        if pos + dl > end:
            return
        yield header, data[pos:pos + dl]
        pos += dl


def chunks(data):
    """Yields (offset, decompressed chunk) for every complete chunk record."""
    marker = b"\x04\x00\x00\x00op=\x05"
    pos = 0
    while True:
        k = data.find(marker, pos)
        if k < 0:
            return
        found = None
        for start in range(max(0, k - 96), k + 1):
            hl = struct.unpack_from("<I", data, start)[0]
            if not 8 <= hl <= 200 or start + 4 + hl + 4 > len(data) or not start + 4 <= k < start + 4 + hl:
                continue
            try:
                h = _fields(bytes(data[start + 4:start + 4 + hl]))
            except (ValueError, UnicodeDecodeError):
                continue
            if h.get("op") == b"\x05" and "compression" in h and "size" in h:
                found = (start, hl, h)
                break
        if found is None:
            pos = k + 1
            continue
        start, hl, h = found
        dl = struct.unpack_from("<I", data, start + 4 + hl)[0]
        body_at = start + 8 + hl
        if body_at + dl > len(data):
            return
        body = data[body_at:body_at + dl]
        comp = h["compression"].decode()
        if comp == "bz2":
            body = bz2.decompress(body)
        elif comp == "lz4":
            import lz4.frame
            body = lz4.frame.decompress(body)
        elif comp != "none":
            raise ValueError(f"unsupported compression {comp}")
        yield start, body
        pos = body_at + dl


def _header(msg, o: int = 0):
    _seq, s, ns, n = struct.unpack_from("<IIII", msg, o)
    return s + ns * 1e-9, o + 16 + n


def _gga(text: str):
    p = text.split(",")
    if len(p) < 10 or not p[0].endswith("GGA") or not p[2] or not p[6]:
        return None
    hh, mm, ss = int(p[1][0:2]), int(p[1][2:4]), float(p[1][4:])
    lat = int(p[2][:2]) + float(p[2][2:]) / 60.
    lat = -lat if p[3] == "S" else lat
    lon = int(p[4][:3]) + float(p[4][3:]) / 60.
    lon = -lon if p[5] == "W" else lon
    return hh * 3600 + mm * 60 + ss, lat, lon, int(p[6]), float(p[9])


def read_ppk(path: Path):
    rows = []
    for line in path.read_text().splitlines():
        if line.startswith("%") or not line.strip():
            continue
        f = line.split()
        y, mo, d = (int(v) for v in f[0].split("/"))
        hh, mm, ss = f[1].split(":")
        gpst = calendar.timegm((y, mo, d, int(hh), int(mm), 0)) + float(ss)
        rows.append((gpst - GPST_MINUS_UTC_S, float(f[2]), float(f[3]), float(f[4]), int(f[5])))
    return np.array(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("slice", type=Path, help="raw byte range of the .bag")
    ap.add_argument("ppk", type=Path, help="PPK .pos (RTKLIB, GPST)")
    ap.add_argument("out", type=Path)
    ap.add_argument("--connections-from", type=Path, required=True,
                    help="first MBs of the same bag: connection records (conn id -> topic) are "
                         "written only in the chunk of first use, not in mid-bag chunks")
    ap.add_argument("--date", default="2022-02-22", help="UTC date of the flight (GGA has time only)")
    ap.add_argument("--scale", type=float, default=0.5, help="output image scale")
    ap.add_argument("--frame-step", type=int, default=2, help="keep every N-th frame (20 Hz source)")
    ap.add_argument("--takeoff-height", type=float, default=None,
                    help="PPK height of the ground at takeoff; default: first PPK epochs")
    args = ap.parse_args()
    frames_dir = args.out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    w, h = int(round(1440 * args.scale)), int(round(1080 * args.scale))
    k_in = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]])
    k_out = np.array([[FX * args.scale, 0, CX * args.scale], [0, FY * args.scale, CY * args.scale], [0, 0, 1]])
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(k_in, np.array(KB), np.eye(3), k_out, (w, h), cv2.CV_16SC2)

    stream = args.slice.open("rb")
    raw = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
    conns: dict[int, str] = {}
    with args.connections_from.open("rb") as head_stream:
        head = mmap.mmap(head_stream.fileno(), 0, access=mmap.ACCESS_READ)
        for _, body in chunks(head):
            for ih, _ in _records(bytes(body), 0, len(body)):
                if ih["op"][0] == 7:
                    conns[struct.unpack("<I", ih["conn"])[0]] = ih["topic"].decode()
            if len(conns) >= 13:
                break
    images, imu, gga = [], [], []
    n_chunks = 0
    for _, body in chunks(raw):
        n_chunks += 1
        body = bytes(body)
        for ih, msg in _records(body, 0, len(body)):
            op = ih["op"][0]
            if op == 7:
                conns[struct.unpack("<I", ih["conn"])[0]] = ih["topic"].decode()
                continue
            if op != 2:
                continue
            topic = conns.get(struct.unpack("<I", ih["conn"])[0])
            s, ns = struct.unpack("<II", ih["time"])
            t_rx = s + ns * 1e-9
            if topic == "/camera/image_mono":
                stamp, o = _header(msg)
                height, width = struct.unpack_from("<II", msg, o)
                o += 8
                n = struct.unpack_from("<I", msg, o)[0]
                enc = msg[o + 4:o + 4 + n].decode()
                o += 4 + n + 1 + 4
                n = struct.unpack_from("<I", msg, o)[0]
                if enc != "mono8" or n != height * width:
                    continue
                images.append((stamp, msg[o + 4:o + 4 + n], height, width))
            elif topic == "/imu/data":
                stamp, o = _header(msg)
                q = struct.unpack_from("<4d", msg, o)
                g = struct.unpack_from("<3d", msg, o + 32 + 72)
                a = struct.unpack_from("<3d", msg, o + 32 + 72 + 24 + 72)
                imu.append((stamp, q, g, a))
            elif topic == "/nmea_sentence":
                _, o = _header(msg)
                n = struct.unpack_from("<I", msg, o)[0]
                parsed = _gga(msg[o + 4:o + 4 + n].decode(errors="replace").strip())
                if parsed:
                    gga.append((t_rx, *parsed))
    if not images or not imu or len(gga) < 5:
        raise SystemExit(f"slice holds too little data: images={len(images)} imu={len(imu)} gga={len(gga)}")

    images.sort(key=lambda r: r[0])
    imu.sort(key=lambda r: r[0])
    day0 = calendar.timegm(tuple(int(v) for v in args.date.split("-")) + (0, 0, 0))
    lat_rx = np.array([r[0] - (day0 + r[1]) for r in gga])
    bag_minus_utc = float(np.min(lat_rx))  # minimum NMEA receive latency
    ppk = read_ppk(args.ppk)
    takeoff_h = args.takeoff_height if args.takeoff_height is not None else float(np.median(ppk[:25, 3]))

    t0 = images[0][0]
    lat0, lon0 = (np.interp(t0 - bag_minus_utc, ppk[:, 0], ppk[:, i]) for i in (1, 2))
    r_m = 6378137.0 * (1 - 0.00669438) / (1 - 0.00669438 * math.sin(math.radians(lat0)) ** 2) ** 1.5
    r_n = 6378137.0 / math.sqrt(1 - 0.00669438 * math.sin(math.radians(lat0)) ** 2)

    def ref(t_bag: float):
        tu = t_bag - bag_minus_utc
        lat, lon, hh, q = (np.interp(tu, ppk[:, 0], ppk[:, i]) for i in (1, 2, 3, 4))
        return (math.radians(lat - lat0) * r_m, math.radians(lon - lon0) * r_n * math.cos(math.radians(lat0)),
                hh - takeoff_h, q)

    def to_us(t: float) -> int:
        return int(round((t - t0) * 1e6)) + 1_000_000

    with (args.out / "manifest.csv").open("w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["timestamp_us", "image_path", "height_agl_m", "true_n_m", "true_e_m",
                     "true_vn_m_s", "true_ve_m_s", "ppk_quality"])
        for i, (stamp, pix, height, width) in enumerate(images[::args.frame_step]):
            img = np.frombuffer(pix, np.uint8).reshape(height, width)
            und = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
            name = f"frames/{i:05d}.png"
            cv2.imwrite(str(args.out / name), und)
            n, e, hagl, q = ref(stamp)
            n1, e1, _, _ = ref(stamp + 0.1)
            n0, e0, _, _ = ref(stamp - 0.1)
            wr.writerow([to_us(stamp), name, f"{hagl:.3f}", f"{n:.4f}", f"{e:.4f}",
                         f"{(n1 - n0) / 0.2:.4f}", f"{(e1 - e0) / 0.2:.4f}", int(round(q))])

    with (args.out / "attitude.csv").open("w", newline="") as fa, \
            (args.out / "imu.csv").open("w", newline="") as fi:
        wa, wi = csv.writer(fa), csv.writer(fi)
        wa.writerow(["timestamp_us", "qw", "qx", "qy", "qz"])
        wi.writerow(["timestamp_us", "gx", "gy", "gz", "ax", "ay", "az"])
        for stamp, (qx, qy, qz, qw), g, a in imu:
            # quaternion (x, y, z, w) -> rotation matrix
            r_enu_b = np.array([
                [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
            ])
            r_nb = R_NED_ENU @ r_enu_b
            qn = _quat_from_matrix(r_nb)
            wa.writerow([to_us(stamp), *(f"{v:.8f}" for v in qn)])
            wi.writerow([to_us(stamp), *(f"{v:.6f}" for v in (*g, *a))])

    rot = BODY_T_CAMDOWN[:, :3]
    calib = {
        "focal_x_px": float(k_out[0, 0]), "focal_y_px": float(k_out[1, 1]),
        "center_x_px": float(k_out[0, 2]), "center_y_px": float(k_out[1, 2]),
        "image_to_body_xy": rot[:2, :2].round(8).tolist(),
        "body_from_camera": rot.round(8).tolist(),
        "camera_offset_body_m": BODY_T_CAMDOWN[:, 3].round(6).tolist(),
    }
    (args.out / "camera.yaml").write_text(json.dumps(calib, indent=2))  # JSON is valid YAML
    span = images[-1][0] - t0
    info = {
        "source": "MUN-FRL VIL dataset (Thalagala et al.), DJI M600, CC BY 4.0",
        "chunks": n_chunks, "frames_total": len(images), "frames_written": len(images[::args.frame_step]),
        "imu_samples": len(imu), "gga": len(gga),
        "gga_fix_quality_counts": {str(q): int(sum(1 for r in gga if r[4] == q)) for q in sorted({r[4] for r in gga})},
        "bag_minus_utc_s": bag_minus_utc, "nmea_receive_latency_spread_s": float(np.ptp(lat_rx)),
        "utc_start": float(t0 - bag_minus_utc - day0), "span_s": span,
        "takeoff_ppk_height_m": takeoff_h,
        "height_agl_range_m": [float(ref(images[0][0])[2]), float(ref(images[-1][0])[2])],
        "height_model": "PPK height minus takeoff height (flat-ground assumption)",
        "camera": {"model": "KANNALA_BRANDT undistorted to pinhole", "scale": args.scale, "size": [w, h]},
    }
    (args.out / "slice_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


def _quat_from_matrix(r: np.ndarray) -> tuple[float, float, float, float]:
    tr = float(np.trace(r))
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = (0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s)
    else:
        i = int(np.argmax(np.diag(r)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(1.0 + r[i, i] - r[j, j] - r[k, k]) * 2
        v = [0.0, 0.0, 0.0]
        v[i] = 0.25 * s
        v[j] = (r[j, i] + r[i, j]) / s
        v[k] = (r[k, i] + r[i, k]) / s
        q = ((r[k, j] - r[j, k]) / s, *v)
    n = math.sqrt(sum(c * c for c in q))
    q = tuple(c / n for c in q)
    return q if q[0] >= 0 else tuple(-c for c in q)


if __name__ == "__main__":
    main()
