"""Geometrically coherent synthetic camera/IMU/range scene and offline replay.

This test bench renders an ideal pinhole nadir camera observing ONE metric
flat orthophoto. Raw FRD gyro/specific-force measurements and the image
projective motion share the same analytic NED trajectory. Camera results
have an explicit arrival delay (not the exposure timestamp). Existing VO,
map-localization and 15D fixed-lag ESKF modules are executed as-is.

This is not PX4 SITL, a photorealistic simulator, a flight controller or
evidence of real 100 m GNSS-denied hovering performance.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from math import cos, sin
from pathlib import Path

import cv2
import numpy as np

from .camera_rotation_compensation import CameraMountCalibration, body_to_ned
from .continuous_visual_odometry import ContinuousPlanarOdometry, VisualFrame
from .error_state_inertial_filter import InertialErrorStateFilter, ImuNoiseDensity
from .fixed_lag_eskf_replay import FixedLagEskfReplay
from .ground_visual_motion import CameraCalibration
from .horizontal_visual_inertial_filter import VisualVelocityMeasurement
from .imu_preintegration import ImuBias, ImuReading
from .orthophoto_localization import MapGeoReference, OrthophotoLocalizer, OrthophotoTile
from .velocity_uncertainty import VelocityErrorAssumptions

SCENARIOS = ("nominal", "yaw_rotation", "camera_blackout",
             "imu_gap", "duplicate_map")


@dataclass(frozen=True)
class BenchConfig:
    scenario: str = "nominal"
    frames: int = 6
    camera_dt_s: float = 0.2
    imu_dt_s: float = 0.01
    height_m: float = 100.0
    seed: int = 72118
    delivery_delay_us: int = 75_000

    def validate(self) -> None:
        if self.scenario not in SCENARIOS:
            raise ValueError("Неизвестный сценарий моделирования")
        if not 4 <= self.frames <= 60:
            raise ValueError("Допустимо от четырех до шестидесяти кадров")
        if abs(self.camera_dt_s / self.imu_dt_s -
               round(self.camera_dt_s / self.imu_dt_s)) > 1e-9:
            raise ValueError("Интервалы камеры и ИИМ должны быть кратны")
        if not (0.08 <= self.camera_dt_s <= 0.3 and
                0.005 <= self.imu_dt_s <= .02 and
                20. <= self.height_m <= 120.):
            raise ValueError("Неверные частоты датчиков или высота")
        if not (0 < self.delivery_delay_us <= 200_000):
            raise ValueError("Неверная синтетическая задержка камеры")
        if not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("Случайная последовательность должна иметь seed >= 0")


def _calibration() -> tuple[CameraCalibration, CameraMountCalibration, MapGeoReference]:
    camera = CameraCalibration(1000., 1000., 320., 240.,
                               ((0., -1.), (1., 0.)))
    mount = CameraMountCalibration(
        ((0., -1., 0.), (1., 0., 0.), (0., 0., 1.))
    )
    georef = MapGeoReference((100., -50.), ((0., -.1), (.1, 0.)))
    assert mount.valid(camera)
    return camera, mount, georef


def _kinematics(t: float, scenario: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Truth pNE, vNE, aNE and analytically matching yaw/angular velocity."""
    position = np.array((
        50. + 1.08*t + .2*sin(.4*t),
        6. + .30*t + .12*sin(.3*t)
    ))
    velocity = np.array((
        1.08+.08*cos(.4*t),
        .30+.036*cos(.3*t)
    ))
    accel = np.array((
        -.032*sin(.4*t),
        -.0108*sin(.3*t)
    ))
    yaw = .06*sin(.8*t) if scenario == "yaw_rotation" else 0.
    yaw_rate = .048*cos(.8*t) if scenario == "yaw_rotation" else 0.
    return position, velocity, accel, yaw, yaw_rate


def _terrain(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = np.zeros((1100, 1280), dtype=np.uint8)
    for _ in range(2900):
        u = int(rng.integers(12, image.shape[1]-12))
        v = int(rng.integers(12, image.shape[0]-12))
        intensity = int(rng.integers(100, 255))
        if rng.random() < .5:
            cv2.circle(image, (u,v), int(rng.integers(2,5)), intensity, -1)
        else:
            cv2.rectangle(image, (u-2, v-2), (u+3,v+3), intensity, 1)
    return image


def render_camera_frame(
    terrain: np.ndarray, position_ne_m: np.ndarray, yaw_rad: float,
    *, height_m: float,
    camera: CameraCalibration, mount: CameraMountCalibration,
    georef: MapGeoReference,
) -> np.ndarray:
    """Analytic projective mapping of each camera ray to a ground PLANE.

    p_ground_ne = p_camera_ne + h * ray_ne / ray_down
    ray_ned = R_nb R_bc K^-1 [u,v,1].
    Homography maps DESTINATION camera pixels -> SOURCE map pixels,
    used with OpenCV WARP_INVERSE_MAP. No arbitrary pixel shifts.
    """
    if terrain.ndim != 2 or terrain.dtype != np.uint8:
        raise ValueError("Неверный растр синтетической местности")
    position = np.asarray(position_ne_m, dtype=float)
    if position.shape != (2,) or not np.isfinite(position).all():
        raise ValueError("Неверная синтетическая позиция камеры")
    if not 2. <= height_m <= 150. or not np.isfinite(yaw_rad):
        raise ValueError("Неверная высота или курс камеры")
    k = np.array([
        [camera.focal_x_px, 0., camera.center_x_px],
        [0., camera.focal_y_px, camera.center_y_px],
        [0.,0.,1.]
    ])
    inv_k = np.linalg.inv(k)
    r_nc = body_to_ned(0.,0.,yaw_rad) @ np.asarray(mount.body_from_camera)
    rays = r_nc @ inv_k
    map_metric = np.linalg.inv(georef.jacobian())
    origin = np.asarray(georef.origin_ne_m)
    down = rays[2,:]
    projected = height_m * map_metric @ rays[:2,:]
    projected += np.outer(map_metric @ (position-origin), down)
    matrix = np.vstack([projected,down])
    frame = cv2.warpPerspective(
        terrain, matrix, (640,480),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    return frame


def generate_dataset(output_dir: str | Path,
                     config: BenchConfig = BenchConfig()) -> Path:
    """Write real PNG frames and CSV measurements from a controlled analytic scene."""
    config.validate()
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    camera, mount, georef = _calibration()
    terrain = _terrain(config.seed)
    if not cv2.imwrite(str(dest/"synthetic_orthophoto.png"),terrain):
        raise OSError("Не удалось записать карту")
    with (dest/"calibration.json").open("w",encoding="utf-8") as f:
        json.dump({
            "camera":asdict(camera),
            "mount":asdict(mount),
            "map_georeference":asdict(georef),
            "scene":"SYNTHETIC_PINHOLE_FLAT_TERRAIN_ONLY",
            "height_reference":"optical centre vertical distance to flat terrain",
        },f,indent=2)

    rng = np.random.default_rng(config.seed+1)
    imu_data = []
    count = int(round((config.frames-1)*config.camera_dt_s/config.imu_dt_s))
    for i in range(count+1):
        t=i*config.imu_dt_s
        position, velocity, accel, yaw, yaw_rate = _kinematics(t,config.scenario)
        rotation=body_to_ned(0.,0.,yaw)
        force_body = rotation.T @ np.array([*accel,-9.80665])
        gyro = np.array((0.,0.,yaw_rate))
        force_body += rng.normal(0., .012, 3)
        gyro += rng.normal(0., .00015, 3)
        time_us=1_000_000+int(round(t*1e6))
        if config.scenario=="imu_gap" and .42 <= t <= .56:
            continue
        imu_data.append([time_us,*gyro.tolist(),*force_body.tolist()])
    with (dest/"imu.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.writer(f)
        writer.writerow(["timestamp_us",
                         "gyro_x_rad_s","gyro_y_rad_s","gyro_z_rad_s",
                         "accel_x_m_s2","accel_y_m_s2","accel_z_m_s2"])
        writer.writerows(imu_data)

    camera_rows=[]
    truth_rows=[]
    for i in range(config.frames):
        t=i*config.camera_dt_s
        position, velocity, accel, yaw, yaw_rate = _kinematics(t,config.scenario)
        timestamp=1_000_000+int(round(t*1e6))
        missing=(config.scenario=="camera_blackout" and i in (2,3))
        image_name=f"frame_{i:03}.png"
        if not missing:
            image=render_camera_frame(
                terrain,position,yaw,height_m=config.height_m,
                camera=camera,mount=mount,georef=georef
            )
            if not cv2.imwrite(str(dest/image_name),image):
                raise OSError("Не удалось записать изображение камеры")
        camera_rows.append([
            timestamp,timestamp+config.delivery_delay_us,
            "" if missing else image_name,0 if missing else 1,
            config.height_m,0.,0.,yaw,timestamp,config.height_m
        ])
        truth_rows.append([
            timestamp,*position.tolist(),*velocity.tolist(),yaw
        ])
    with (dest/"camera.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.writer(f)
        writer.writerow([
            "timestamp_us","arrival_timestamp_imu_us","image_path","available",
            "height_agl_m","roll_rad","pitch_rad","yaw_rad",
            "range_timestamp_us","range_height_m"
        ])
        writer.writerows(camera_rows)
    with (dest/"truth.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.writer(f)
        writer.writerow(["timestamp_us","north_m","east_m","vn_m_s","ve_m_s","yaw_rad"])
        writer.writerows(truth_rows)
    with (dest/"bench_metadata.json").open("w",encoding="utf-8") as f:
        json.dump({
            "config":asdict(config),
            "synthetic_only":True,
            "map_file":"synthetic_orthophoto.png",
            "camera_file":"camera.csv",
            "imu_file":"imu.csv",
            "range_source":"synthetic camera optical-centre height",
            "truth_file":"truth.csv",
            "ground_truth_used_by_estimators":False,
            "px4_control_enabled":False,
        },f,ensure_ascii=False,indent=2)
    return dest


def _imu_rows(path: Path) -> list[ImuReading]:
    with path.open(newline="",encoding="utf-8") as f:
        rows=list(csv.DictReader(f))
    return [
        ImuReading(
            int(r["timestamp_us"]),
            tuple(float(r[f"gyro_{a}_rad_s"]) for a in "xyz"),
            tuple(float(r[f"accel_{a}_m_s2"]) for a in "xyz"),
        ) for r in rows
    ]


def run_sensor_bench(data_dir: str | Path,
                     output_dir: str | Path) -> dict:
    """Exercise VO, orthophoto and delayed ESKF with shared frame/IMU timeline.

    Reference truth is read only AFTER all state estimates were produced.
    This function is intentionally not a command-and-control loop.
    """
    base=Path(data_dir)
    metadata=json.loads((base/"bench_metadata.json").read_text(encoding="utf-8"))
    if metadata.get("synthetic_only") is not True:
        raise ValueError("Стенд принимает только явно маркированную синтетику")
    config=BenchConfig(**metadata["config"])
    config.validate()
    camera,mount,georef=_calibration()
    with (base/"camera.csv").open(newline="",encoding="utf-8") as f:
        camera_rows=list(csv.DictReader(f))
    imu=_imu_rows(base/"imu.csv")
    if not imu or not camera_rows:
        raise ValueError("Нет синтетических измерений")
    tile=cv2.imread(str(base/"synthetic_orthophoto.png"),cv2.IMREAD_GRAYSCALE)
    if tile is None:
        raise ValueError("Отсутствует ортофотоплан")
    tiles=[OrthophotoTile("reference",tile,georef)]
    if config.scenario=="duplicate_map":
        tiles.append(OrthophotoTile(
            "ambiguous_duplicate",tile.copy(),
            MapGeoReference((400.,900.),georef.matrix_ne_m_per_px)
        ))
    localizer=OrthophotoLocalizer(tiles,camera,mount)
    vo=ContinuousPlanarOdometry(
        camera,camera_mount=mount,
        velocity_error_assumptions=VelocityErrorAssumptions(
            .55,.35,.001,.003,.001,.02,.12
        )
    )
    def factory():
        return InertialErrorStateFilter(
            quaternion_body_to_ned=(1.,0.,0.,0.),
            initial_bias=ImuBias(),
            noise=ImuNoiseDensity(.003,.03,.00005,.0003),
            max_imu_dt_s=.05,
        )
    fusion=FixedLagEskfReplay(factory,max_lag_us=300_000,
                              max_imu_samples=100)
    timeline=[(r.timestamp_us,0,"imu",r) for r in imu]
    timeline.extend((
        int(row["arrival_timestamp_imu_us"]),1,"camera",row
    ) for row in camera_rows)
    timeline.sort(key=lambda e:(e[0],e[1]))
    results=[]
    imu_failed=False
    last_camera_time=0
    for event_time,_,kind,payload in timeline:
        if kind=="imu":
            if not imu_failed:
                try:
                    fusion.push_imu(payload)
                except ValueError:
                    imu_failed=True
            continue
        row=payload
        stamp=int(row["timestamp_us"])
        if stamp<=last_camera_time:
            raise ValueError("Неверный порядок экспозиций камеры")
        last_camera_time=stamp
        if int(row["available"])==0:
            observation=vo.reject_frame(stamp,"SYNTHETIC_CAMERA_UNAVAILABLE")
            map_status="CAMERA_UNAVAILABLE"
            map_ne=(float("nan"),float("nan"))
        else:
            image=cv2.imread(str(base/row["image_path"]),cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError("Не найден кадр из манифеста")
            # Synthetic range timestamp refers to exposure time exactly.
            if int(row["range_timestamp_us"])!=stamp:
                raise ValueError("Дальность и экспозиция не синхронизированы")
            observation=vo.process(VisualFrame(
                stamp,image,float(row["range_height_m"]),
                float(row["yaw_rad"]),float(row["roll_rad"]),
                float(row["pitch_rad"])
            ))
            fix=localizer.localize(
                image,height_agl_m=float(row["height_agl_m"]),
                roll_rad=float(row["roll_rad"]),
                pitch_rad=float(row["pitch_rad"]),
                yaw_rad=float(row["yaw_rad"]),
            )
            map_status=fix.status
            map_ne=(fix.north_m,fix.east_m)
        fusion_status="IMU_FAILURE_LATCHED" if imu_failed else "VO_NOT_ACCEPTED"
        if not imu_failed and observation.accepted:
            if not np.all(np.isfinite([
                observation.vn_m_s,observation.ve_m_s,
                observation.velocity_cov_nn,observation.velocity_cov_ne,
                observation.velocity_cov_ee
            ])):
                fusion_status="VO_NO_VALID_COVARIANCE"
            else:
                measured=VisualVelocityMeasurement(
                    stamp,observation.vn_m_s,observation.ve_m_s,
                    ((observation.velocity_cov_nn,observation.velocity_cov_ne),
                     (observation.velocity_cov_ne,observation.velocity_cov_ee))
                )
                fusion_status=fusion.push_delayed_visual(measured).status
        state=fusion.state
        results.append({
            "timestamp_us":stamp,
            "arrival_timestamp_imu_us":event_time,
            "camera_available":row["available"],
            "visual_status":observation.status,
            "visual_segment_id":observation.segment_id,
            "visual_velocity_accepted":int(observation.accepted),
            "vn_vo_m_s":observation.vn_m_s,
            "ve_vo_m_s":observation.ve_m_s,
            "map_status":map_status,
            "map_n_m":map_ne[0],
            "map_e_m":map_ne[1],
            "eskf_status":fusion_status,
            "eskf_vn_m_s":state.velocity_ned_m_s[0],
            "eskf_ve_m_s":state.velocity_ned_m_s[1],
            "applied_map_correction":0,
        })
    # Independent synthetic truth opened only for scoring, after estimating.
    with (base/"truth.csv").open(newline="",encoding="utf-8") as f:
        truth={int(r["timestamp_us"]):r for r in csv.DictReader(f)}
    map_errors=[]
    vo_errors=[]
    fusion_errors=[]
    for row in results:
        ref=truth[row["timestamp_us"]]
        if row["map_status"]=="GEOMETRICALLY_ACCEPTED_UNVALIDATED":
            map_errors.append(float(np.hypot(
                row["map_n_m"]-float(ref["north_m"]),
                row["map_e_m"]-float(ref["east_m"])
            )))
        if row["visual_velocity_accepted"]:
            vo_errors.append(float(np.hypot(
                row["vn_vo_m_s"]-float(ref["vn_m_s"]),
                row["ve_vo_m_s"]-float(ref["ve_m_s"])
            )))
        if row["eskf_status"]=="VISUAL_CORRECTED":
            fusion_errors.append(float(np.hypot(
                row["eskf_vn_m_s"]-float(ref["vn_m_s"]),
                row["eskf_ve_m_s"]-float(ref["ve_m_s"])
            )))
    def rmse(errors):
        return float(np.sqrt(np.mean(np.square(errors)))) if errors else None
    summary={
        "scenario":config.scenario,
        "frames":len(results),
        "camera_available":sum(int(r["camera_available"]) for r in results),
        "visual_velocity_accepted":sum(r["visual_velocity_accepted"] for r in results),
        "map_geometry_accepted":len(map_errors),
        "visual_updates_fused":sum(r["eskf_status"]=="VISUAL_CORRECTED" for r in results),
        "map_rejected_ambiguity":sum(r["map_status"]=="AMBIGUOUS_MAP_MATCH" for r in results),
        "imu_failure_latched":imu_failed,
        "map_position_rmse_m":rmse(map_errors),
        "visual_speed_rmse_m_s":rmse(vo_errors),
        "eskf_speed_rmse_m_s":rmse(fusion_errors),
        "applied_map_correction":False,
        "flight_accuracy_validated":False,
        "kind":"IDEAL_PINHOLE_SYNTHETIC_SENSOR_BENCH",
        "limitations":[
            "Идеальная плоская местность и проектная камера; не моделируются реальные аберрации, рельеф и скользящий затвор",
            "ИИМ, диапазон высоты, курс и геопривязка созданы по одной модельной траектории",
            "Эталон прочитан только для оценки результата, после завершения работы навигационных компонентов",
            "Ковариация VO основана на заданных предположениях, а не на метрологической проверке",
            "Коррекция карты не применяется к ESKF и PX4, контур управления отсутствует",
            "Нет летных записей НИИВК и нет подтверждения работы реального БВС на 100 м",
        ],
    }
    dest=Path(output_dir)
    dest.mkdir(parents=True,exist_ok=True)
    with (dest/"sensor_bench_results.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    with (dest/"sensor_bench_summary.json").open("w",encoding="utf-8") as f:
        json.dump(summary,f,ensure_ascii=False,indent=2,allow_nan=False)
    return summary


def main() -> None:
    parser=argparse.ArgumentParser(
        description="Синтетические изображения камеры, ИИМ, дальномер, VO, карта и отложенный РФК"
    )
    parser.add_argument("--output-dir",required=True)
    parser.add_argument("--scenario",choices=SCENARIOS,default="nominal")
    parser.add_argument("--frames",type=int,default=6)
    parser.add_argument("--seed",type=int,default=72118)
    args=parser.parse_args()
    config=BenchConfig(scenario=args.scenario,frames=args.frames,seed=args.seed)
    base=generate_dataset(Path(args.output_dir)/"sensors",config)
    result=run_sensor_bench(base,Path(args.output_dir)/"evaluation")
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
