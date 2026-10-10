"""Compare C++ and Python 15D ESKF on the SAME deterministic IMU+late-video log.

Standalone CI scientific regression, never an onboard application.
Checks entire 225 covariance entries, not only VN/VE.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np

from src.error_state_inertial_filter import InertialErrorStateFilter, ImuNoiseDensity
from src.fixed_lag_eskf_replay import FixedLagEskfReplay
from src.horizontal_visual_inertial_filter import VisualVelocityMeasurement
from src.imu_preintegration import ImuBias, ImuReading


def run_parity(executable: str | Path, *, atol: float = 1.e-6) -> dict:
    result = subprocess.run(
        [str(Path(executable).resolve())],
        check=True, capture_output=True, text=True, timeout=20,
    )
    cpp = json.loads(result.stdout)
    python = FixedLagEskfReplay(
        lambda: InertialErrorStateFilter(
            quaternion_body_to_ned=(1.,0.,0.,0.),
            initial_bias=ImuBias(),
            noise=ImuNoiseDensity(.003,.03,.00005,.0003),
            max_imu_dt_s=.05,
        ),
        max_lag_us=500000,
        max_imu_samples=64,
    )
    for i in range(50):
        t=i*.01
        python.push_imu(ImuReading(
            1000000+i*10000,
            (0.,0.,.02),
            (
                .1*math.sin(.37*t),
                .02*math.cos(.31*t),
                -9.80665+.02*math.sin(.41*t),
            )
        ))
    for timestamp,vn,ve in ((1255000,.13,.03),(1120000,.07,.02)):
        feedback=python.push_delayed_visual(
            VisualVelocityMeasurement(
                timestamp,vn,ve,((.04,0.),(0.,.04))
            )
        )
        if feedback.status != "VISUAL_CORRECTED":
            raise AssertionError(
                f"Python delayed visual failed: {feedback.status}"
            )
    reference=python.state
    expected={
        "position":np.asarray(reference.position_ned_m,dtype=float),
        "velocity":np.asarray(reference.velocity_ned_m_s,dtype=float),
        "quaternion":np.asarray(reference.quaternion_body_to_ned,dtype=float),
        "gyro_bias":np.asarray(reference.gyro_bias_rad_s,dtype=float),
        "accel_bias":np.asarray(reference.accel_bias_m_s2,dtype=float),
        "covariance":np.asarray(reference.covariance_15x15,dtype=float).ravel(),
    }
    tolerances={}
    for field,vector in expected.items():
        actual=np.asarray(cpp[field],dtype=float)
        if actual.shape!=vector.shape or not np.isfinite(actual).all():
            raise AssertionError(f"Invalid C++ {field} vector")
        abs_error=np.abs(actual-vector)
        error=float(np.max(abs_error))
        tolerances[field]=error
        np.testing.assert_allclose(
            actual, vector, rtol=1.e-7, atol=atol,
            err_msg=f"Full C++/Python ESKF parity: {field}"
        )
    if cpp["timestamp_us"] != reference.timestamp_us:
        raise AssertionError("C++ and Python final timestamps disagree")
    if cpp["accepted"] != reference.accepted_visual_count:
        raise AssertionError("C++ and Python accepted counts disagree")
    if cpp["rejected"] != reference.rejected_visual_count:
        raise AssertionError("C++ and Python rejected counts disagree")
    nis_error=abs(cpp["nis"]-reference.last_nis)
    if nis_error>atol:
        raise AssertionError("C++ and Python NIS disagree")
    return {
        "type":"DETERMINISTIC_SYNTHETIC_CPP_PYTHON_ESKF_PARITY",
        "passed":True,
        "state_dimensions":15,
        "covariance_entries_compared":225,
        "imu_samples":50,
        "out_of_order_visual_measurements":2,
        "one_visual_time_interpolated_between_imu_samples":True,
        "timestamp_us":cpp["timestamp_us"],
        "max_absolute_error_by_field":tolerances,
        "last_nis_absolute_error":nis_error,
        "tolerance_absolute":atol,
        "validated_for_flight":False,
        "no_px4_ekf2_fusion":True,
    }


def main() -> None:
    parser=argparse.ArgumentParser(description="Сравнение Python и C++ 15-мерного РФК")
    parser.add_argument("--cpp-executable",required=True)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    report=run_parity(args.cpp_executable)
    out=Path(args.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
