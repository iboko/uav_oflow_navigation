import math

from src.config import NavigationConfig
from src.oflow_math import optical_flow_to_velocity, velocity_to_optical_flow_rad


def test_velocity_to_flow_and_back():
    cfg = NavigationConfig()
    vx_nav = 0.2
    vy_nav = -0.1
    yaw = 0.3
    height = 1.2
    dt = 0.04
    gx = 0.02
    gy = -0.01
    gz = 0.005

    ix, iy, ixg, iyg, izg = velocity_to_optical_flow_rad(
        vx_nav, vy_nav, yaw, height, dt, gx, gy, gz, flow_noise_rad=0.0
    )

    res = optical_flow_to_velocity(
        integrated_x_rad=ix,
        integrated_y_rad=iy,
        integrated_xgyro_rad=ixg,
        integrated_ygyro_rad=iyg,
        dt_s=dt,
        range_m=height,
        roll_rad=0.0,
        pitch_rad=0.0,
        yaw_rad=yaw,
        quality=200,
        cfg=cfg,
    )

    assert res.valid
    assert abs(res.vx_nav_mps - vx_nav) < 1e-9
    assert abs(res.vy_nav_mps - vy_nav) < 1e-9
