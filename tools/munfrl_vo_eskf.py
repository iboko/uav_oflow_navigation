#!/usr/bin/env python3
"""VO + 15-мерный РФК на реальном срезе MUN-FRL против PPK.

Вход — каталог импортера (manifest.csv, imu.csv, attitude.csv) и вывод
src.run_camera_odometry (visual_odometry.csv). Один и тот же РФК
(src.fixed_lag_eskf_replay, запаздывающие измерения с пересчетом)
прогоняется в режимах:

  imu        — только ИИМ (нижняя граница: дрейф без коррекции);
  vo         — коррекция скоростью VO N, E с R = (h·σrate)²·I;
  vo_scaled  — то же, но скорость VO поделена на масштаб полета по PPK
               (заменитель дальномера: допущение плоской земли снято).

Метки ИИМ в мешке — время приема пакетами (интервалы 0,05–30 мс при
среднем 400 Гц без пропусков): восстанавливается равномерная сетка по
нижней огибающей (задержка передачи неотрицательна), затем усреднение
до 100 Гц. Ориентация и курс РФК в начале — от Xsens (та же, что
использует VO), поэтому для сравнения с PPK оценка поворачивается на
один угол полета; угол и масштаб определяются по первой половине
среза, метрики считаются по второй.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.error_state_inertial_filter import ImuNoiseDensity, InertialErrorStateFilter  # noqa: E402
from src.fixed_lag_eskf_replay import FixedLagEskfReplay  # noqa: E402
from src.horizontal_visual_inertial_filter import VisualVelocityMeasurement  # noqa: E402
from src.imu_preintegration import ImuBias, ImuReading  # noqa: E402

# MUN-FRL calibration page (VINS-style continuous densities, Xsens MTi).
IMU_NOISE = ImuNoiseDensity(gyro_rad_s_sqrt_hz=0.004, accel_m_s2_sqrt_hz=0.08,
                            gyro_bias_walk_rad_s2_sqrt_hz=1e-4,
                            accel_bias_walk_m_s3_sqrt_hz=4e-5)
MODES = ("imu", "vo", "vo_scaled")


def dejitter(stamps: np.ndarray) -> tuple[np.ndarray, dict]:
    """Равномерная сетка t = a + b·i по нижней огибающей меток приема."""
    i = np.arange(len(stamps), dtype=float)
    b, a = np.polyfit(i, stamps.astype(float), 1)
    a += float(np.min(stamps - (a + b * i)))
    fitted = a + b * i
    lag = stamps - fitted
    return fitted, {"rate_hz": 1e6 / b, "receive_lag_us_p50": float(np.median(lag)),
                    "receive_lag_us_p99": float(np.percentile(lag, 99))}


def load_imu(path: Path, decimate: int = 4) -> tuple[list[ImuReading], dict]:
    d = np.loadtxt(path, delimiter=",", skiprows=1)
    if d.ndim != 2 or d.shape[1] != 7 or len(d) < 10 * decimate:
        raise ValueError("Недостаточно отсчетов ИИМ")
    t, info = dejitter(d[:, 0])
    n = (len(d) // decimate) * decimate
    blocks = d[:n, 1:].reshape(-1, decimate, 6).mean(axis=1)
    ends = t[decimate - 1:n:decimate]
    readings = [ImuReading(int(round(te)), tuple(map(float, row[:3])), tuple(map(float, row[3:])))
                for te, row in zip(ends, blocks)]
    info["output_rate_hz"] = info["rate_hz"] / decimate
    return readings, info


def load_attitude(path: Path, t_us: int) -> tuple[float, float, float, float]:
    rows = np.loadtxt(path, delimiter=",", skiprows=1)
    k = int(np.argmin(np.abs(rows[:, 0] - t_us)))
    q = rows[k, 1:5]
    return tuple(float(x) for x in q / np.linalg.norm(q))


def load_frames(manifest: Path, vo_csv: Path) -> list[dict]:
    with manifest.open(newline="", encoding="utf-8-sig") as fp:
        man = list(csv.DictReader(fp))
    index = {r["timestamp_us"]: i for i, r in enumerate(man)}
    frames = []
    with vo_csv.open(newline="", encoding="utf-8-sig") as fp:
        for r in csv.DictReader(fp):
            i = index[r["timestamp_us"]]
            row = man[i]
            frames.append({
                "t": int(row["timestamp_us"]),
                "t_prev": int(man[i - 1]["timestamp_us"]) if i else None,
                "accepted": r["accepted"] == "1" and i > 0,
                "vo": (float(r["vn_m_s"]), float(r["ve_m_s"])),
                "h": float(row["height_agl_m"]),
                "ref_pos": (float(row["true_n_m"]), float(row["true_e_m"])),
                "ref_vel": (float(row["true_vn_m_s"]), float(row["true_ve_m_s"])),
                "ppk_fix": str(row.get("ppk_quality", "1")).strip() == "1",
            })
    return frames


def procrustes(src: np.ndarray, dst: np.ndarray) -> tuple[float, float]:
    """dst ≈ s·R(θ)·src по МНК; возвращает (θ, s)."""
    cross = float(np.sum(src[:, 0] * dst[:, 1] - src[:, 1] * dst[:, 0]))
    dot = float(np.sum(src[:, 0] * dst[:, 0] + src[:, 1] * dst[:, 1]))
    return math.atan2(cross, dot), math.hypot(cross, dot) / float(np.sum(src * src))


def rot(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def run_mode(mode: str, imu: list[ImuReading], frames: list[dict], q0, sigma_rate: float,
             vo_scale: float) -> dict:
    if mode not in MODES:
        raise ValueError(mode)
    sig = (10., 10., 10., 5., 5., 5., 0.05, 0.05, 0.2, 0.01, 0.01, 0.01, 0.3, 0.3, 0.3)

    def factory():
        return InertialErrorStateFilter(quaternion_body_to_ned=q0, initial_bias=ImuBias(),
                                        noise=IMU_NOISE, initial_error_sigma=sig)

    replay = FixedLagEskfReplay(factory, max_lag_us=300_000, max_imu_samples=200)
    k, out, nis, statuses = 0, [], [], {}
    for f in frames:
        while k < len(imu) and imu[k].timestamp_us <= f["t"]:
            replay.push_imu(imu[k])
            k += 1
        if mode != "imu" and f["accepted"]:
            mid = (f["t"] + f["t_prev"]) // 2
            vn, ve = (f["vo"][0] / vo_scale, f["vo"][1] / vo_scale) if mode == "vo_scaled" else f["vo"]
            var = (f["h"] * sigma_rate) ** 2
            if mode == "vo_scaled":
                var /= vo_scale ** 2
            res = replay.push_delayed_visual(VisualVelocityMeasurement(
                mid, vn, ve, ((var, 0.0), (0.0, var))))
            statuses[res.status] = statuses.get(res.status, 0) + 1
            if res.status == "VISUAL_CORRECTED" and math.isfinite(res.nis):
                nis.append(res.nis)
        s = replay.state
        out.append({"t": f["t"], "p": s.position_ned_m[:2], "v": s.velocity_ned_m_s[:2],
                    "pv": (s.covariance_15x15[3][3], s.covariance_15x15[4][4])})
    return {"trajectory": out, "statuses": statuses,
            "nis_mean": float(np.mean(nis)) if nis else None, "nis_count": len(nis)}


def evaluate(frames: list[dict], traj: list[dict], theta: float) -> dict:
    """Метрики второй половины среза в осях PPK после поворота на θ."""
    r = rot(theta)
    n = len(frames)
    half = n // 2
    est_v = np.array([r @ np.array(x["v"]) for x in traj])
    ref_v = np.array([f["ref_vel"] for f in frames])
    ok = np.array([f["ppk_fix"] for f in frames])
    sel = np.arange(n) >= half
    err = (est_v - ref_v)[sel & ok]
    est_p = np.array([r @ np.array(x["p"]) for x in traj])
    ref_p = np.array([f["ref_pos"] for f in frames])
    # Drift relative to the start of the validation half.
    d_est = est_p[sel] - est_p[half]
    d_ref = ref_p[sel] - ref_p[half]
    drift = np.linalg.norm(d_est - d_ref, axis=1)
    path = float(np.sum(np.linalg.norm(np.diff(ref_p[sel], axis=0), axis=1)))
    sigma_v = np.sqrt(np.array([x["pv"] for x in traj])[sel & ok])
    return {
        "velocity_rmse_m_s": float(np.sqrt(np.mean(np.sum(err ** 2, axis=1)))),
        "velocity_mean_error_ne_m_s": err.mean(axis=0).round(3).tolist(),
        "velocity_nees_2d": float(np.mean(np.sum((err / sigma_v) ** 2, axis=1))),
        "final_drift_m": float(drift[-1]),
        "max_drift_m": float(drift.max()),
        "path_length_m": path,
        "final_drift_percent_of_path": 100.0 * float(drift[-1]) / path if path > 0 else None,
        "duration_s": (frames[-1]["t"] - frames[half]["t"]) * 1e-6,
    }


def vo_dead_reckoning(frames: list[dict], vo_scale: float) -> list[dict]:
    p = np.zeros(2)
    out = []
    for f in frames:
        if f["accepted"]:
            p = p + np.array(f["vo"]) / vo_scale * (f["t"] - f["t_prev"]) * 1e-6
        out.append({"t": f["t"], "p": tuple(p), "v": tuple(np.array(f["vo"]) / vo_scale)
                    if f["accepted"] else (float("nan"),) * 2, "pv": (1.0, 1.0)})
    return out


def run(slice_dir: Path, vo_csv: Path, sigma_rate: float) -> dict:
    imu, imu_info = load_imu(slice_dir / "imu.csv")
    frames = load_frames(slice_dir / "manifest.csv", vo_csv)
    frames = [f for f in frames if imu[0].timestamp_us < f["t"] <= imu[-1].timestamp_us]
    half = len(frames) // 2
    q0 = load_attitude(slice_dir / "attitude.csv", imu[0].timestamp_us)
    # Heading of the Xsens frame and VO speed scale from the FIRST half only.
    acc = [f for f in frames[:half] if f["accepted"] and f["ppk_fix"]
           and np.hypot(*f["ref_vel"]) > 1.0]
    if len(acc) < 10:
        raise ValueError("Недостаточно движения в первой половине среза для выставки")
    theta_vo, scale = procrustes(np.array([f["vo"] for f in acc]), np.array([f["ref_vel"] for f in acc]))
    vo_scale = 1.0 / scale  # vo ≈ vo_scale · R(-θ) · ref
    report = {"imu": imu_info, "frames": len(frames), "vo_heading_offset_deg": -math.degrees(theta_vo),
              "vo_speed_scale": vo_scale, "sigma_rate_rad_s": sigma_rate, "modes": {}}
    for mode in MODES:
        res = run_mode(mode, imu, frames, q0, sigma_rate, vo_scale)
        traj = res["trajectory"]
        # The ESKF frame shares the Xsens heading: one rotation from the first half.
        est = np.array([x["v"] for x in traj[:half]])
        ref = np.array([f["ref_vel"] for f in frames[:half]])
        theta, _ = procrustes(est, ref)
        report["modes"][mode] = {"statuses": res["statuses"], "nis_mean": res["nis_mean"],
                                 "nis_count": res["nis_count"], "heading_alignment_deg": math.degrees(theta),
                                 **evaluate(frames, traj, theta)}
    for name, scale_used in (("vo_dead_reckoning", 1.0), ("vo_scaled_dead_reckoning", vo_scale)):
        traj = vo_dead_reckoning(frames, scale_used)
        m = evaluate(frames, traj, theta_vo)
        report["modes"][name] = {k: m[k] for k in ("final_drift_m", "max_drift_m", "path_length_m",
                                                   "final_drift_percent_of_path", "duration_s")}
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("slices", nargs="+", help="NAME:SLICE_DIR:VO_CSV")
    ap.add_argument("--sigma-rate", type=float, default=0.028,
                    help="σrate, рад/с (калибровка munfrl_flight_calibration)")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    report = {"type": "MUNFRL_VO_ESKF_VS_PPK", "covariance_applied_to_px4": False, "flights": {}}
    for spec in args.slices:
        name, slice_dir, vo = spec.split(":")
        report["flights"][name] = run(Path(slice_dir), Path(vo), args.sigma_rate)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        args.out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
