from __future__ import annotations

import argparse
from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from config import NavigationConfig
    from ekf_oflow import HorizontalOpticalFlowEKF
    from oflow_math import optical_flow_to_velocity, altitude_from_range
    from safety_monitor import OpticalFlowSafetyMonitor
except ImportError:
    from .config import NavigationConfig
    from .ekf_oflow import HorizontalOpticalFlowEKF
    from .oflow_math import optical_flow_to_velocity, altitude_from_range
    from .safety_monitor import OpticalFlowSafetyMonitor


def run_demo(input_csv: str | Path, outdir: str | Path, cfg: NavigationConfig) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ekf = HorizontalOpticalFlowEKF(cfg)
    monitor = OpticalFlowSafetyMonitor(cfg)

    rows: list[dict] = []
    status_count: dict[str, int] = {}

    last_range_m = math.nan

    for _, r in df.iterrows():
        t_s = float(r["t_s"])
        roll = float(r["roll_rad"])
        pitch = float(r["pitch_rad"])
        yaw = float(r["yaw_rad"])

        acc_x = float(r["acc_x_mps2"])
        acc_y = float(r["acc_y_mps2"])
        imu_valid = np.isfinite(acc_x) and np.isfinite(acc_y) and np.isfinite(yaw)

        if imu_valid:
            ekf.predict(t_s, acc_x, acc_y, yaw)
        elif not ekf.initialized:
            ekf.initialize(t_s)

        range_received = np.isfinite(r.get("range_m", np.nan))
        if range_received:
            last_range_m = float(r["range_m"])

        height_m = altitude_from_range(last_range_m, roll, pitch)

        flow_received = np.isfinite(r.get("of_integrated_x_rad", np.nan))
        flow_quality = float(r["of_quality"]) if np.isfinite(r.get("of_quality", np.nan)) else math.nan

        status = monitor.update(
            t_s=t_s,
            range_m=last_range_m,
            height_m=height_m,
            flow_quality=flow_quality,
            imu_valid=imu_valid,
            flow_received=flow_received,
            range_received=range_received,
        )
        status_count[status.mode] = status_count.get(status.mode, 0) + 1

        flow_valid = False
        flow_reason = "NO_FLOW_SAMPLE"
        vflow_n = math.nan
        vflow_e = math.nan
        innov_vn = math.nan
        innov_ve = math.nan

        if flow_received:
            fres = optical_flow_to_velocity(
                integrated_x_rad=float(r["of_integrated_x_rad"]),
                integrated_y_rad=float(r["of_integrated_y_rad"]),
                integrated_xgyro_rad=float(r["of_integrated_xgyro_rad"]),
                integrated_ygyro_rad=float(r["of_integrated_ygyro_rad"]),
                dt_s=float(r["of_dt_s"]),
                range_m=last_range_m,
                roll_rad=roll,
                pitch_rad=pitch,
                yaw_rad=yaw,
                quality=flow_quality,
                cfg=cfg,
            )
            flow_valid = fres.valid and status.allow_optical_flow
            flow_reason = fres.reason
            if fres.valid:
                vflow_n = fres.vx_nav_mps
                vflow_e = fres.vy_nav_mps
                if status.allow_optical_flow:
                    innov_vn, innov_ve = ekf.update_flow_velocity(
                        vn_meas_mps=fres.vx_nav_mps,
                        ve_meas_mps=fres.vy_nav_mps,
                        quality=flow_quality,
                        height_m=fres.height_m,
                    )

        st = ekf.state(t_s)
        rows.append(
            {
                "t_s": t_s,
                "est_n_m": st.n_m,
                "est_e_m": st.e_m,
                "est_vn_mps": st.vn_mps,
                "est_ve_mps": st.ve_mps,
                "est_bax_mps2": st.bax_mps2,
                "est_bay_mps2": st.bay_mps2,
                "height_used_m": height_m,
                "flow_vn_mps": vflow_n,
                "flow_ve_mps": vflow_e,
                "flow_valid": int(flow_valid),
                "flow_reason": flow_reason,
                "safety_mode": status.mode,
                "warnings": ",".join(status.warnings),
                "innov_vn_mps": innov_vn,
                "innov_ve_mps": innov_ve,
                "true_n_m": float(r["true_n_m"]),
                "true_e_m": float(r["true_e_m"]),
                "true_vn_mps": float(r["true_vn_mps"]),
                "true_ve_mps": float(r["true_ve_mps"]),
                "true_alt_m": float(r["true_alt_m"]),
                "of_quality": flow_quality,
                "texture_class": r["texture_class"],
            }
        )

    out = pd.DataFrame(rows)
    out_csv = outdir / "estimated_nav.csv"
    out.to_csv(out_csv, index=False)

    pos_err = np.hypot(out["est_n_m"] - out["true_n_m"], out["est_e_m"] - out["true_e_m"])
    vel_err = np.hypot(out["est_vn_mps"] - out["true_vn_mps"], out["est_ve_mps"] - out["true_ve_mps"])

    summary = {
        "rows": int(len(out)),
        "position_rmse_m": float(np.sqrt(np.mean(pos_err**2))),
        "position_p95_m": float(np.percentile(pos_err, 95)),
        "velocity_rmse_mps": float(np.sqrt(np.mean(vel_err**2))),
        "velocity_p95_mps": float(np.percentile(vel_err, 95)),
        "flow_valid_samples": int(out["flow_valid"].sum()),
        "status_count": status_count,
    }
    with (outdir / "status_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    plt.figure(figsize=(7, 6))
    plt.plot(out["true_e_m"], out["true_n_m"], label="истинная траектория")
    plt.plot(out["est_e_m"], out["est_n_m"], label="оценка EKF")
    plt.xlabel("E, м")
    plt.ylabel("N, м")
    plt.title("Относительная траектория БВС")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "trajectory_xy.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(out["t_s"], out["true_vn_mps"], label="true Vn")
    plt.plot(out["t_s"], out["est_vn_mps"], label="EKF Vn")
    plt.plot(out["t_s"], out["true_ve_mps"], label="true Ve")
    plt.plot(out["t_s"], out["est_ve_mps"], label="EKF Ve")
    plt.xlabel("t, с")
    plt.ylabel("скорость, м/с")
    plt.title("Горизонтальные скорости")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "velocity.png", dpi=180)
    plt.close()

    print(f"Сохранено: {out_csv}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Офлайн-демонстрация ОП-навигации БВС")
    parser.add_argument("--input", default="data/synthetic_oflow_log.csv", help="Входной CSV")
    parser.add_argument("--outdir", default="outputs", help="Каталог результатов")
    parser.add_argument("--config", default=None, help="YAML-конфигурация")
    args = parser.parse_args()

    cfg = NavigationConfig.from_yaml(args.config) if args.config else NavigationConfig()
    run_demo(args.input, args.outdir, cfg)


if __name__ == "__main__":
    main()
