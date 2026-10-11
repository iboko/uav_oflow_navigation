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
from .error_state_inertial_filter import (
    BodyVelocityMeasurement, ImuNoiseDensity, InertialErrorStateFilter,
    MapGroundPointMeasurement, YawMeasurement,
)
from .fixed_lag_eskf_replay import FixedLagEskfReplay
from .ground_visual_motion import CameraCalibration
from .horizontal_visual_inertial_filter import VisualVelocityMeasurement
from .imu_preintegration import (
    ImuBias, ImuReading, quat_exponential, quat_product, rotation_body_to_ned,
)
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
    truth_imu_rows = []
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
        truth_imu_rows.append([time_us,*velocity.tolist(),*position.tolist(),yaw])
        if config.scenario=="imu_gap" and .42 <= t <= .56:
            continue
        imu_data.append([time_us,*gyro.tolist(),*force_body.tolist()])
    with (dest/"imu.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.writer(f)
        writer.writerow(["timestamp_us",
                         "gyro_x_rad_s","gyro_y_rad_s","gyro_z_rad_s",
                         "accel_x_m_s2","accel_y_m_s2","accel_z_m_s2"])
        writer.writerows(imu_data)
    with (dest/"truth_imu.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.writer(f)
        writer.writerow(["timestamp_us","vn_m_s","ve_m_s","north_m","east_m","yaw_rad"])
        writer.writerows(truth_imu_rows)

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
            "truth_at_imu_times_file":"truth_imu.csv",
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


def _rpy(q: tuple[float, float, float, float]) -> tuple[float, float, float]:
    w, x, y, z = q
    roll = float(np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y)))
    pitch = float(np.arcsin(np.clip(2*(w*y-z*x), -1., 1.)))
    yaw = float(np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z)))
    return roll, pitch, yaw


def _wrap(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def _gyro_delta(imu: list[ImuReading], t0: int, t1: int,
                bias: np.ndarray) -> np.ndarray:
    """Body rotation quaternion from t0 to t1 by bias-corrected gyro only.

    Trapezoidal rates between samples, linear interpolation at the ends.
    This is the PHYSICAL relative rotation; differences of corrected
    filter attitudes also contain estimation jumps and must not be used
    to derotate image pairs.
    """
    if t1 < t0:
        raise ValueError("Обратный интервал интегрирования гироскопа")
    stamps = [r.timestamp_us for r in imu]
    if not stamps or t0 < stamps[0] or t1 > stamps[-1]:
        raise ValueError("Интервал вне журнала ИИМ")
    def rate(t: int) -> np.ndarray:
        j = int(np.searchsorted(stamps, t))
        if j < len(stamps) and stamps[j] == t:
            return np.asarray(imu[j].gyro_body_rad_s) - bias
        a, b = imu[j-1], imu[j]
        alpha = (t - a.timestamp_us) / (b.timestamp_us - a.timestamp_us)
        return ((1-alpha) * np.asarray(a.gyro_body_rad_s) +
                alpha * np.asarray(b.gyro_body_rad_s)) - bias
    knots = [t0] + [t for t in stamps if t0 < t < t1] + [t1]
    q = np.array([1., 0., 0., 0.])
    for a, b in zip(knots, knots[1:]):
        omega = 0.5 * (rate(a) + rate(b))
        q = quat_product(q, quat_exponential((b - a) * 1e-6 * omega))
    return q / np.linalg.norm(q)


def load_bench_calibration(path: str | Path, *, evaluation_seed: int) -> dict:
    """Calibration from SEPARATE synthetic flights (src.synthetic_bench_calibration)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("kind") != "SYNTHETIC_BENCH_ERROR_CALIBRATION" or data.get("synthetic_only") is not True:
        raise ValueError("Калибровка не является калибровкой синтетического стенда")
    seeds = set(data["seeds"]["fit"]) | set(data["seeds"]["holdout"])
    if evaluation_seed in seeds:
        raise ValueError("Оцениваемый прогон использован при калибровке")
    if not data["vo_body_velocity"]["consistent_on_holdout"]:
        raise ValueError("Калибровка ковариации VO не подтверждена на отложенных полетах")
    return data


def run_sensor_bench(data_dir: str | Path,
                     output_dir: str | Path,
                     *, calibration: dict | None = None) -> dict:
    """Exercise VO, orthophoto and delayed ESKF as a loosely coupled VIO.

    The ESKF's OWN attitude (FixedLagEskfReplay.attitude_at) drives VO
    derotation and orthophoto matching; the autopilot attitude columns of
    camera.csv are ignored. VO aids the filter by body-frame velocity at
    the midpoint of the frame interval. With a calibration from separate
    flights the orthophoto heading and position (covariance intersection,
    unknown VO correlation) are fused too. Reference truth is read only
    AFTER all estimates were produced. Never a command-and-control loop.
    """
    base=Path(data_dir)
    metadata=json.loads((base/"bench_metadata.json").read_text(encoding="utf-8"))
    if metadata.get("synthetic_only") is not True:
        raise ValueError("Стенд принимает только явно маркированную синтетику")
    config=BenchConfig(**metadata["config"])
    config.validate()
    camera,mount,georef=_calibration()
    # The generated calibration is a recorded contract for this bench.
    # Never silently ignore a changed calibration while using hardcoded
    # simulator parameters: it would invalidate all metric comparisons.
    recorded=json.loads((base/"calibration.json").read_text(encoding="utf-8"))
    expected={
        "camera":json.loads(json.dumps(asdict(camera))),
        "mount":json.loads(json.dumps(asdict(mount))),
        "map_georeference":json.loads(json.dumps(asdict(georef))),
    }
    if any(recorded.get(name)!=value for name,value in expected.items()):
        raise ValueError("Записанная калибровка противоречит модели синтетического стенда")
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
    map_cal=calibration["map_ground_point"] if calibration else None
    yaw_cal=calibration["map_yaw"] if calibration else None
    vo_cal=calibration["vo_body_velocity"] if calibration else None
    initial_q=(1.,0.,0.,0.)
    # Initial horizontal position: first unambiguous orthophoto fix of the
    # first exposure (coincident with the first IMU sample), otherwise the
    # relative origin. Uncalibrated fixes start with a 10 m sigma.
    initial_ne=(0.,0.)
    anchored=False
    first=camera_rows[0]
    if int(first["available"])==1 and int(first["timestamp_us"])==imu[0].timestamp_us:
        image=cv2.imread(str(base/first["image_path"]),cv2.IMREAD_GRAYSCALE)
        if image is not None:
            seed_fix=localizer.localize(image,height_agl_m=float(first["range_height_m"]),
                                        roll_rad=0.,pitch_rad=0.,yaw_rad=0.)
            if seed_fix.accepted:
                # Level-attitude assumption at start: optical-axis ground
                # point is directly below the vehicle.
                initial_ne=(seed_fix.ground_north_m,seed_fix.ground_east_m)
                anchored=True
                if map_cal:
                    initial_ne=(initial_ne[0]-map_cal["bias_ne_m"][0],
                                initial_ne[1]-map_cal["bias_ne_m"][1])
    position_sigma=10.
    if map_cal:
        position_sigma=float(np.sqrt(np.max(np.linalg.eigvalsh(
            np.asarray(map_cal["covariance_ne_m2"])))))
    sigma=(position_sigma,position_sigma,10.,1.,1.,1.,.05,.05,.05,
           .02,.02,.02,.2,.2,.2)
    def factory():
        return InertialErrorStateFilter(
            quaternion_body_to_ned=initial_q,
            initial_bias=ImuBias(),
            noise=ImuNoiseDensity(.003,.03,.00005,.0003),
            initial_error_sigma=sigma,
            max_imu_dt_s=.05,
            initial_position_ned_m=(initial_ne[0],initial_ne[1],0.),
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
    nan=float("nan")
    # Attitude chain handed to VO: identity at segment start, then
    # propagated by gyro only, so pair derotation is physical and the
    # body-frame velocity never contains the filter's attitude estimate.
    vo_q=None
    vo_time=0
    vo_previous_q=None
    # FIXED gyro bias of the initial alignment/calibration for derotation.
    # The running filter estimate is not used: the body-velocity model has
    # no bias Jacobian, so feeding it back would be an unmodelled loop.
    vo_gyro_bias=np.asarray(ImuBias().gyro_body_rad_s)
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
        body=(nan,nan,nan,nan,nan,0)
        vo_ne=None
        map_status="CAMERA_UNAVAILABLE"
        map_ne=(nan,nan)
        ground_ne=(nan,nan)
        map_yaw=nan
        rpy=(nan,nan,nan)
        vo_rpy=(nan,nan,nan)
        try:
            if not imu_failed:
                q_now=fusion.attitude_at(stamp)
                rpy=_rpy(q_now)
                bias=vo_gyro_bias
                if vo_q is None:
                    # Segment-local frame: the first frame of a segment
                    # defines the VO reference axes. Anchoring to the filter
                    # attitude would feed the filter's own tilt error back
                    # into the "measurement" and destabilize the loop.
                    vo_q=np.array([1.,0.,0.,0.])
                else:
                    vo_q=quat_product(vo_q,_gyro_delta(imu,vo_time,stamp,bias))
                vo_rpy=_rpy(tuple(vo_q))
        except ValueError:
            vo_q=None
        if int(row["available"])==0:
            observation=vo.reject_frame(stamp,"SYNTHETIC_CAMERA_UNAVAILABLE")
        elif not np.all(np.isfinite(rpy)):
            observation=vo.reject_frame(stamp,"ESKF_ATTITUDE_UNAVAILABLE")
            map_status="ESKF_ATTITUDE_UNAVAILABLE"
        else:
            image=cv2.imread(str(base/row["image_path"]),cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError("Не найден кадр из манифеста")
            # Synthetic range timestamp refers to exposure time exactly.
            if int(row["range_timestamp_us"])!=stamp:
                raise ValueError("Дальность и экспозиция не синхронизированы")
            if abs(float(row["range_height_m"])-float(row["height_agl_m"])) > .05:
                raise ValueError("Дальномер и высота геометрической модели не согласованы")
            observation=vo.process(VisualFrame(
                stamp,image,float(row["range_height_m"]),
                vo_rpy[2],vo_rpy[0],vo_rpy[1]
            ))
            fix=localizer.localize(
                image,height_agl_m=float(row["range_height_m"]),
                roll_rad=rpy[0],pitch_rad=rpy[1],yaw_rad=rpy[2],
            )
            map_status=fix.status
            map_ne=(fix.north_m,fix.east_m)
            ground_ne=(fix.ground_north_m,fix.ground_east_m)
            map_yaw=fix.yaw_rad
        chain_previous=vo_previous_q if observation.accepted else None
        if vo_q is not None:
            if observation.status in ("VALID","INITIALIZED"):
                vo_previous_q=np.asarray(vo_q)
                vo_time=stamp
            else:
                # Segment restarted: next pair re-anchors to the filter.
                vo_q=None
                vo_previous_q=None
        vo_status="IMU_FAILURE_LATCHED" if imu_failed else "VO_NOT_ACCEPTED"
        if not imu_failed and observation.accepted:
            if not np.all(np.isfinite([
                observation.vn_m_s,observation.ve_m_s,
                observation.velocity_cov_nn,observation.velocity_cov_ne,
                observation.velocity_cov_ee
            ])) or observation.interval_start_us<=0:
                vo_status="VO_NO_VALID_COVARIANCE"
            else:
                # Mean velocity over the pair -> body axes at the interval
                # midpoint with the ESKF's own attitude there. The absolute
                # heading cancels: VO used the same filter attitude to form NE.
                t_mid=(observation.interval_start_us+stamp)//2
                # Same chained attitude VO used, at the midpoint: the
                # result depends on image and gyro only, not on the
                # absolute filter heading.
                q_mid=quat_product(np.asarray(chain_previous),_gyro_delta(
                    imu,observation.interval_start_us,t_mid,vo_gyro_bias))
                r_mid=rotation_body_to_ned(q_mid)
                r2=r_mid[:2,:2]
                v_body=r2.T@np.array([observation.vn_m_s,observation.ve_m_s])
                c_ne=np.array([[observation.velocity_cov_nn,observation.velocity_cov_ne],
                               [observation.velocity_cov_ne,observation.velocity_cov_ee]])
                c_body=r2.T@c_ne@r2
                if vo_cal:
                    v_body=v_body-np.asarray(vo_cal["bias_m_s"])
                    c_body=vo_cal["scale"]*c_body+np.eye(2)*vo_cal["floor_variance_m2_s2"]
                c_body=0.5*(c_body+c_body.T)
                body=(float(v_body[0]),float(v_body[1]),float(c_body[0,0]),
                      float(c_body[0,1]),float(c_body[1,1]),t_mid)
                # NE velocity reported for diagnostics/export only, rotated
                # with the filter attitude at the midpoint (not fused).
                r_eskf=rotation_body_to_ned(fusion.attitude_at(t_mid))
                vo_ne=r_eskf[:2,:2]@np.array(body[:2])
                vo_status=fusion.push_delayed_frame([BodyVelocityMeasurement(
                    t_mid,body[0],body[1],((body[2],body[3]),(body[3],body[4]))
                )])[0].status
        map_fusion_status="MAP_COVARIANCE_UNVALIDATED"
        yaw_fusion_status="MAP_YAW_UNVALIDATED"
        if map_status!="GEOMETRICALLY_ACCEPTED_UNVALIDATED":
            map_fusion_status=yaw_fusion_status=map_status
        elif imu_failed:
            map_fusion_status=yaw_fusion_status="IMU_FAILURE_LATCHED"
        elif map_cal and yaw_cal:
            # Attitude-free ground point of the optical axis; the filter
            # model p + h (R d)_NE/(R d)_D couples it to the tilt states.
            optical_axis=tuple(float(x) for x in
                               np.asarray(mount.body_from_camera)@np.array([0.,0.,1.]))
            measurements=[
                YawMeasurement(stamp,_wrap(map_yaw-yaw_cal["bias_rad"]),
                               yaw_cal["variance_rad2"]),
                MapGroundPointMeasurement(
                    stamp,ground_ne[0]-map_cal["bias_ne_m"][0],
                    ground_ne[1]-map_cal["bias_ne_m"][1],
                    tuple(tuple(x) for x in map_cal["covariance_ne_m2"]),
                    float(row["range_height_m"]),optical_axis,0.0),
            ]
            yaw_feedback,map_feedback=fusion.push_delayed_frame(measurements)
            yaw_fusion_status,map_fusion_status=yaw_feedback.status,map_feedback.status
        state=fusion.state
        results.append({
            "timestamp_us":stamp,
            "arrival_timestamp_imu_us":event_time,
            "camera_available":row["available"],
            "eskf_attitude_roll_rad":rpy[0],
            "eskf_attitude_pitch_rad":rpy[1],
            "eskf_attitude_yaw_rad":rpy[2],
            "visual_status":observation.status,
            "visual_segment_id":observation.segment_id,
            "visual_velocity_accepted":int(observation.accepted),
            "vn_vo_m_s":float(vo_ne[0]) if vo_ne is not None else observation.vn_m_s,
            "ve_vo_m_s":float(vo_ne[1]) if vo_ne is not None else observation.ve_m_s,
            "body_time_us":body[5],
            "body_vx_vo_m_s":body[0],
            "body_vy_vo_m_s":body[1],
            "body_cov_xx":body[2],
            "body_cov_xy":body[3],
            "body_cov_yy":body[4],
            "map_status":map_status,
            "map_n_m":map_ne[0],
            "map_e_m":map_ne[1],
            "map_yaw_rad":map_yaw,
            "map_ground_n_m":ground_ne[0],
            "map_ground_e_m":ground_ne[1],
            "eskf_status":vo_status,
            "eskf_yaw_update_status":yaw_fusion_status,
            "eskf_map_update_status":map_fusion_status,
            "eskf_vn_m_s":state.velocity_ned_m_s[0],
            "eskf_ve_m_s":state.velocity_ned_m_s[1],
            "eskf_n_m":state.position_ned_m[0],
            "eskf_e_m":state.position_ned_m[1],
            "eskf_yaw_rad":_rpy(state.quaternion_body_to_ned)[2],
            "eskf_state_timestamp_us":state.timestamp_us,
            "applied_map_correction":int(map_fusion_status=="POSITION_CORRECTED"),
        })
    # Independent synthetic truth opened only for scoring, after estimating.
    with (base/"truth.csv").open(newline="",encoding="utf-8") as f:
        truth={int(r["timestamp_us"]):r for r in csv.DictReader(f)}
    with (base/"truth_imu.csv").open(newline="",encoding="utf-8") as f:
        truth_imu={int(r["timestamp_us"]):r for r in csv.DictReader(f)}
    start_truth=truth_imu[min(truth_imu)]
    start_n,start_e=float(start_truth["north_m"]),float(start_truth["east_m"])
    def truth_at(t_us: int):
        position,velocity,_,yaw,_=_kinematics((t_us-1_000_000)*1e-6,config.scenario)
        return position,velocity,yaw
    map_errors=[]
    map_yaw_errors=[]
    vo_errors=[]
    body_errors=[]
    fusion_errors=[]
    position_errors=[]
    yaw_errors=[]
    for row in results:
        ref=truth[row["timestamp_us"]]
        if row["map_status"]=="GEOMETRICALLY_ACCEPTED_UNVALIDATED":
            map_errors.append(float(np.hypot(
                row["map_n_m"]-float(ref["north_m"]),
                row["map_e_m"]-float(ref["east_m"])
            )))
            map_yaw_errors.append(abs(_wrap(row["map_yaw_rad"]-float(ref["yaw_rad"]))))
        if row["visual_velocity_accepted"]:
            vo_errors.append(float(np.hypot(
                row["vn_vo_m_s"]-float(ref["vn_m_s"]),
                row["ve_vo_m_s"]-float(ref["ve_m_s"])
            )))
        if np.isfinite(row["body_vx_vo_m_s"]):
            _,v_true,yaw_true=truth_at(int(row["body_time_us"]))
            c,s_=np.cos(yaw_true),np.sin(yaw_true)
            body_true=np.array([c*v_true[0]+s_*v_true[1],-s_*v_true[0]+c*v_true[1]])
            body_errors.append(float(np.hypot(row["body_vx_vo_m_s"]-body_true[0],
                                              row["body_vy_vo_m_s"]-body_true[1])))
        if row["eskf_status"]=="VISUAL_CORRECTED":
            # Delayed correction is applied at exposure time, but the
            # replayed filter snapshot is at the latest received IMU time.
            # Comparing it with exposure-time truth would be a false error.
            at_state=truth_imu.get(row["eskf_state_timestamp_us"])
            if at_state is None:
                raise ValueError("Отсутствует эталон на времени оценки РФК")
            fusion_errors.append(float(np.hypot(
                row["eskf_vn_m_s"]-float(at_state["vn_m_s"]),
                row["eskf_ve_m_s"]-float(at_state["ve_m_s"])
            )))
            # Without a map anchor the filter frame starts at the origin:
            # score displacement from the start instead of absolute position.
            ref_n=float(at_state["north_m"])-(0. if anchored else start_n)
            ref_e=float(at_state["east_m"])-(0. if anchored else start_e)
            position_errors.append(float(np.hypot(
                row["eskf_n_m"]-ref_n,row["eskf_e_m"]-ref_e
            )))
            yaw_errors.append(abs(_wrap(row["eskf_yaw_rad"]-float(at_state["yaw_rad"]))))
    def rmse(errors):
        return float(np.sqrt(np.mean(np.square(errors)))) if errors else None
    applied=sum(r["applied_map_correction"] for r in results)
    summary={
        "scenario":config.scenario,
        "frames":len(results),
        "camera_available":sum(int(r["camera_available"]) for r in results),
        "visual_velocity_accepted":sum(r["visual_velocity_accepted"] for r in results),
        "map_geometry_accepted":len(map_errors),
        "visual_updates_fused":sum(r["eskf_status"]=="VISUAL_CORRECTED" for r in results),
        "map_rejected_ambiguity":sum(r["map_status"]=="AMBIGUOUS_MAP_MATCH" for r in results),
        "imu_failure_latched":imu_failed,
        "attitude_source":"ESKF_INTERNAL",
        "vo_aiding":"BODY_FRAME_VELOCITY_AT_INTERVAL_MIDPOINT",
        "map_calibration_used":calibration is not None,
        "map_yaw_updates_fused":sum(r["eskf_yaw_update_status"]=="YAW_CORRECTED" for r in results),
        "map_position_updates_fused":applied,
        "map_position_fusion":("GROUND_POINT_FULL_STATE_COVARIANCE_INTERSECTION"
                               if calibration else "NOT_APPLIED_UNCALIBRATED"),
        "position_reference":"MAP" if anchored else "RELATIVE_TO_START",
        "map_position_rmse_m":rmse(map_errors),
        "map_yaw_error_max_rad":max(map_yaw_errors) if map_yaw_errors else None,
        "visual_speed_rmse_m_s":rmse(vo_errors),
        "vo_body_velocity_rmse_m_s":rmse(body_errors),
        "eskf_speed_rmse_m_s":rmse(fusion_errors),
        "eskf_position_rmse_m":rmse(position_errors),
        "eskf_yaw_error_max_rad":max(yaw_errors) if yaw_errors else None,
        "applied_map_correction":applied>0,
        "flight_accuracy_validated":False,
        "kind":"IDEAL_PINHOLE_SYNTHETIC_SENSOR_BENCH",
        "limitations":[
            "Идеальная плоская местность и проектная камера; не моделируются реальные аберрации, рельеф и скользящий затвор",
            "ИИМ, диапазон высоты, курс и геопривязка созданы по одной модельной траектории",
            "Эталон прочитан только для оценки результата, после завершения работы навигационных компонентов",
            "Ковариации VO и карты откалиброваны только на других СИНТЕТИЧЕСКИХ полетах" if calibration else "Ковариация VO основана на заданных предположениях; карта не применяется без калибровки",
            "Слабосвязанная VIO: VO использует ориентацию самого РФК, но характерные точки не входят в состояние",
            "Контур управления отсутствует; результаты не передаются в PX4/EKF2",
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
    parser.add_argument("--calibration",
                        help="JSON из src.synthetic_bench_calibration (другие seed)")
    args=parser.parse_args()
    config=BenchConfig(scenario=args.scenario,frames=args.frames,seed=args.seed)
    calibration=(load_bench_calibration(args.calibration,evaluation_seed=args.seed)
                 if args.calibration else None)
    base=generate_dataset(Path(args.output_dir)/"sensors",config)
    result=run_sensor_bench(base,Path(args.output_dir)/"evaluation",
                            calibration=calibration)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
