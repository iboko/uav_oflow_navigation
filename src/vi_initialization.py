"""Инициализация VIO без ГНСС: линейная выставка по признакам и ИИМ.

Замкнутая форма (Martinelli, 2014; Kaiser и др., 2017) для ориентации,
известной от AHRS (крен и тангаж по гравитации, курс произвольный).
Начало координат — положение на первом кадре окна, p_0 = 0. Для кадра k

    p_k = v0·t_k + ½·g·t_k² + α_k − A_k·b_a,

где α_k = ∬R·f — двойной интеграл удельной силы, A_k = ∬R, R — ориентация
по гироскопам от ориентации AHRS. Центр камеры c_k = p_k + R_k·p_bc,
луч наблюдения b_ik = R_k·R_bc·K⁻¹[u, v, 1]. Условие X_i = c_k + λ·b_ik
исключением λ ([b]× (X_i − c_k) = 0) линейно по неизвестным
(v0, b_a, X_i). Один МНК с весами по оцененной дальности дает оценку и
ее ковариацию.

Масштаб наблюдаем только при ускорениях: при равномерном движении
система вырождена. Решение принимается, только если СКО скорости и
смещения меньше порогов. Иначе окно надо удлинить (``initialize`` это
сообщает), а не запускать VIO с непроверенной скоростью.

Высота, дальномер и ГНСС не используются.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .error_state_inertial_filter import skew
from .imu_preintegration import (
    GRAVITY_NED, ImuReading, _unit_quaternion, quat_exponential, quat_product, rotation_body_to_ned,
)
from .msckf_vio import PinholeCamera


@dataclass(frozen=True)
class ViInitResult:
    ok: bool
    status: str
    t_end_us: int = 0
    velocity_end_ned_m_s: tuple[float, float, float] = (float("nan"),) * 3
    velocity_end_cov: tuple = ()
    accel_bias_body_m_s2: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quaternion_end: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    velocity0_ned_m_s: tuple[float, float, float] = (float("nan"),) * 3
    sigma_velocity_m_s: float = float("nan")
    sigma_accel_bias_m_s2: float = float("nan")
    features: int = 0
    observations: int = 0
    median_depth_m: float = float("nan")
    chi2_per_dof: float = float("nan")
    gravity_tilt_deg: tuple[float, float] = (0.0, 0.0)
    excitation_m_s2: float = float("nan")


class _Preintegrator:
    """Nominal R, ∫R·f, ∬R·f, ∫R, ∬R from the window start (midpoint rule)."""

    def __init__(self, q0, gyro_bias=(0.0, 0.0, 0.0)) -> None:
        self.q = _unit_quaternion(q0)
        self.bg = np.asarray(gyro_bias, dtype=float)
        self.beta = np.zeros(3)     # ∫ R f
        self.alpha = np.zeros(3)    # ∬ R f
        self.b_int = np.zeros((3, 3))   # ∫ R
        self.a_int = np.zeros((3, 3))   # ∬ R
        self.t = 0.0
        self.last: ImuReading | None = None
        self.acc_world: list[np.ndarray] = []   # R f + g per step (excitation)

    def push(self, r: ImuReading) -> None:
        if self.last is None:
            self.last = r
            return
        dt = (r.timestamp_us - self.last.timestamp_us) * 1e-6
        if not 0 < dt <= 0.05:
            raise ValueError("Разрыв ИИМ в окне инициализации")
        w = 0.5 * (np.asarray(r.gyro_body_rad_s) + np.asarray(self.last.gyro_body_rad_s)) - self.bg
        f = 0.5 * (np.asarray(r.specific_force_body_m_s2) + np.asarray(self.last.specific_force_body_m_s2))
        rmid = rotation_body_to_ned(_unit_quaternion(quat_product(self.q, quat_exponential(0.5 * dt * w))))
        a = rmid @ f
        self.acc_world.append(a + GRAVITY_NED)
        self.alpha += self.beta * dt + 0.5 * a * dt * dt
        self.beta += a * dt
        self.a_int += self.b_int * dt + 0.5 * rmid * dt * dt
        self.b_int += rmid * dt
        self.q = _unit_quaternion(quat_product(self.q, quat_exponential(w * dt)))
        self.t += dt
        self.last = r


def _rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(a @ b)
    vx = skew(v)
    return np.eye(3) + vx + vx @ vx / (1.0 + c)


def _quat_from_matrix(r: np.ndarray) -> np.ndarray:
    w = np.sqrt(max(1e-12, 1.0 + r[0, 0] + r[1, 1] + r[2, 2])) / 2.0
    return _unit_quaternion(np.array([w, (r[2, 1] - r[1, 2]) / (4 * w), (r[0, 2] - r[2, 0]) / (4 * w),
                                      (r[1, 0] - r[0, 1]) / (4 * w)]))


def _median_depth(sol, tracks, states, nb, p_bc) -> float:
    d = []
    for j, (_, obs) in enumerate(tracks.items()):
        tk, rk, alpha, a_int = states[obs[0][0]]
        c = sol[0:3] * tk + 0.5 * GRAVITY_NED * tk * tk + alpha + rk @ p_bc
        if nb:
            c = c - a_int @ sol[3:6]
        d.append(float(np.linalg.norm(sol[3 + nb + 3 * j:6 + nb + 3 * j] - c)))
    return float(np.median(d))


def _with_samples_at(imu: list[ImuReading], times: list[int]) -> list[ImuReading] | None:
    """IMU readings plus linearly interpolated readings exactly at the frame times."""
    stamps = [r.timestamp_us for r in imu]
    out = list(imu)
    extra = []
    for t in times:
        i = int(np.searchsorted(stamps, t))
        if i < len(stamps) and stamps[i] == t:
            continue
        if i == 0 or i == len(stamps):
            return None
        a, b = imu[i - 1], imu[i]
        w = (t - a.timestamp_us) / (b.timestamp_us - a.timestamp_us)
        mix = lambda x, y: tuple((1 - w) * np.asarray(x) + w * np.asarray(y))  # noqa: E731
        extra.append(ImuReading(t, mix(a.gyro_body_rad_s, b.gyro_body_rad_s),
                                mix(a.specific_force_body_m_s2, b.specific_force_body_m_s2)))
    out.extend(extra)
    out.sort(key=lambda r: r.timestamp_us)
    return out


def initialize(imu: list[ImuReading], frames: list[tuple[int, dict[int, tuple[float, float]]]],
               camera: PinholeCamera, q0, *, gyro_bias=(0.0, 0.0, 0.0), estimate_accel_bias: bool = True,
               min_track: int = 4, max_sigma_v: float = 0.25, max_sigma_ba: float = 0.15,
               accel_bias_prior_sigma: float = 0.1, max_chi2_per_dof: float = 3.0,
               heights_agl_m: dict[int, float] | None = None, tilt_prior_sigma_rad: float = 0.035,
               height_sigma_fraction: float = 0.05, min_excitation_m_s2: float = 0.3) -> ViInitResult:
    """frames: [(t_us, {id: (u, v)})] with t_us on the IMU clock, in order;
    imu must cover [frames[0].t, frames[-1].t]; q0 = AHRS attitude at frames[0].t.

    heights_agl_m: optional {t_us: height above flat ground} from a
    rangefinder. Each feature is then a ground point: its depth below the
    camera of frame k is the measured height (sigma = fraction x height).
    Without it the scale comes from accelerations only.
    """
    if len(frames) < 4:
        return ViInitResult(False, "TOO_FEW_FRAMES")
    r_bc, p_bc = camera.matrices()
    k_inv = np.linalg.inv(np.array([[camera.fx, 0, camera.cx], [0, camera.fy, camera.cy], [0, 0, 1.0]]))
    t0, t_last = frames[0][0], frames[-1][0]
    pre = _Preintegrator(q0, gyro_bias)
    readings = _with_samples_at(imu, [t for t, _ in frames])
    if readings is None:
        return ViInitResult(False, "IMU_DOES_NOT_COVER_WINDOW")
    readings = [r for r in readings if t0 <= r.timestamp_us <= t_last]
    states = {}
    ri = 0
    for t_us, _ in frames:
        while ri < len(readings) and readings[ri].timestamp_us <= t_us:
            pre.push(readings[ri])
            ri += 1
        states[t_us] = (pre.t, rotation_body_to_ned(pre.q), pre.alpha.copy(), pre.a_int.copy())
    tracks: dict[int, list[tuple[int, float, float]]] = {}
    for t_us, feats in frames:
        for fid, (u, v) in feats.items():
            tracks.setdefault(fid, []).append((t_us, u, v))
    tracks = {i: o for i, o in tracks.items() if len(o) >= min_track}
    if len(tracks) < 10:
        return ViInitResult(False, "TOO_FEW_TRACKS", features=len(tracks))
    nfeat = len(tracks)
    sigma_theta = camera.pixel_sigma / camera.fx

    def solve(nb: int, weights: dict | None):
        n = 3 + nb + 3 * nfeat
        rows, rhs, wts = [], [], []
        for j, (fid, obs) in enumerate(tracks.items()):
            for t_us, u, v in obs:
                tk, rk, alpha, a_int = states[t_us]
                b = rk @ r_bc @ (k_inv @ np.array([u, v, 1.0]))
                b /= np.linalg.norm(b)
                sb = skew(b)
                row = np.zeros((3, n))
                row[:, 0:3] = -sb * tk
                if nb:
                    row[:, 3:6] = sb @ a_int
                row[:, 3 + nb + 3 * j:6 + nb + 3 * j] = sb
                rows.append(row)
                rhs.append(sb @ (0.5 * GRAVITY_NED * tk * tk + alpha + rk @ p_bc))
                wts.append(1.0 if weights is None else weights[fid])
        a_mat = np.vstack(rows)
        y = np.concatenate(rhs)
        w_rows = np.repeat(np.asarray(wts), 3)
        aw, yw = a_mat * w_rows[:, None], y * w_rows
        if nb:
            # Prior b_a ~ N(0, prior²): over a short window with nearly
            # constant acceleration the bias trades against the scale.
            prior = np.zeros((3, n))
            prior[:, 3:6] = np.eye(3) / accel_bias_prior_sigma
            aw, yw = np.vstack([aw, prior]), np.concatenate([yw, np.zeros(3)])
        sol, *_ = np.linalg.lstsq(aw, yw, rcond=None)
        return sol, aw, len(rows)

    # Pass 1: no bias, unweighted -> ranges of the features.
    sol1, _, _ = solve(0, None)
    weights, depths = {}, []
    for j, (fid, obs) in enumerate(tracks.items()):
        x = sol1[3 + 3 * j:6 + 3 * j]
        tk, rk, alpha, _ = states[obs[0][0]]
        c = sol1[0:3] * tk + 0.5 * GRAVITY_NED * tk * tk + alpha + rk @ p_bc
        d = float(np.linalg.norm(x - c))
        depths.append(d)
        weights[fid] = 1.0 / max(d * sigma_theta, 1e-4)
    if not np.median(depths) > 0.5:
        return ViInitResult(False, "DEGENERATE_GEOMETRY", features=nfeat)
    # Linear solution only as the starting point: noise in the bearings
    # multiplies the unknown points (errors in variables), which biases the
    # cross-product least squares towards a smaller scene and, with a free
    # accelerometer bias, collapses the scale. Maximum likelihood is the
    # pixel reprojection error: Gauss-Newton over (v0, b_a, X_i).
    nb = 3 if estimate_accel_bias else 0
    # Two tilt angles of the world frame (gravity direction) at the end of
    # the state: an AHRS tilt error of 1 deg leaks 0.17 m/s^2 of gravity,
    # 0.3 m/s of velocity over 2 s.
    ng = 3 + nb + 3 * nfeat
    n = ng + 2
    g_jac = -skew(GRAVITY_NED)[:, :2]   # d(g + phi x g)/d(phi_x, phi_y)

    def gravity(x):
        return GRAVITY_NED + g_jac @ x[ng:ng + 2]

    sol = np.zeros(n)
    sol[0:3] = sol1[0:3]
    sol[3 + nb:ng] = sol1[3:]
    obs_list = [(j, t_us, u, v) for j, (fid, o) in enumerate(tracks.items()) for t_us, u, v in o]
    nobs = len(obs_list)
    r_cb = r_bc.T
    # One height constraint per feature, at its first observation.
    height_rows = [] if not heights_agl_m else [
        (j, o[0][0]) for j, o in enumerate(tracks.values()) if o[0][0] in heights_agl_m]
    if heights_agl_m:
        # Start from the height scale instead of the (possibly collapsed)
        # linear one: scale the linear solution so the median depth matches.
        ratio = float(np.median([heights_agl_m[t] for _, t in height_rows])) / float(np.median(depths))
        if ratio > 0:
            sol[0:3] *= ratio
            for j, (_, obs) in enumerate(tracks.items()):
                tk, rk, alpha, _ = states[obs[0][0]]
                c = 0.5 * GRAVITY_NED * tk * tk + alpha + rk @ p_bc
                sol[3 + nb + 3 * j:6 + nb + 3 * j] = c + ratio * (sol[3 + nb + 3 * j:6 + nb + 3 * j] - c)
    for _ in range(8):
        jac = np.zeros((2 * nobs + nb + len(height_rows) + 2, n))
        res = np.zeros(2 * nobs + nb + len(height_rows) + 2)
        for m, (j, t_us, u, v) in enumerate(obs_list):
            tk, rk, alpha, a_int = states[t_us]
            c = sol[0:3] * tk + 0.5 * gravity(sol) * tk * tk + alpha + rk @ p_bc
            if nb:
                c = c - a_int @ sol[3:6]
            r_cw = r_cb @ rk.T
            pc = r_cw @ (sol[3 + nb + 3 * j:6 + nb + 3 * j] - c)
            if pc[2] <= 0.1:
                return ViInitResult(False, "POINT_BEHIND_CAMERA", features=nfeat, observations=nobs)
            jp = np.array([[camera.fx / pc[2], 0.0, -camera.fx * pc[0] / pc[2] ** 2],
                           [0.0, camera.fy / pc[2], -camera.fy * pc[1] / pc[2] ** 2]]) / camera.pixel_sigma
            res[2 * m:2 * m + 2] = (np.array([u, v]) - np.array([camera.fx * pc[0] / pc[2] + camera.cx,
                                                                   camera.fy * pc[1] / pc[2] + camera.cy])
                                    ) / camera.pixel_sigma
            jw = jp @ r_cw
            jac[2 * m:2 * m + 2, 0:3] = -jw * tk
            if nb:
                jac[2 * m:2 * m + 2, 3:6] = jw @ a_int
            jac[2 * m:2 * m + 2, 3 + nb + 3 * j:6 + nb + 3 * j] = jw
            jac[2 * m:2 * m + 2, ng:ng + 2] = -jw @ g_jac * (0.5 * tk * tk)
        if nb:
            # Prior b_a ~ N(0, prior^2).
            jac[2 * nobs:2 * nobs + 3, 3:6] = np.eye(3) / accel_bias_prior_sigma
            res[2 * nobs:2 * nobs + 3] = -sol[3:6] / accel_bias_prior_sigma
        # Prior on the tilt: N(0, tilt_prior^2) per axis.
        tilt_row = 2 * nobs + nb + len(height_rows)
        jac[tilt_row:tilt_row + 2, ng:ng + 2] = np.eye(2) / tilt_prior_sigma_rad
        res[tilt_row:tilt_row + 2] = -sol[ng:ng + 2] / tilt_prior_sigma_rad
        for m, (j, t_us) in enumerate(height_rows):
            tk, rk, alpha, a_int = states[t_us]
            c = sol[0:3] * tk + 0.5 * gravity(sol) * tk * tk + alpha + rk @ p_bc
            if nb:
                c = c - a_int @ sol[3:6]
            sig_h = height_sigma_fraction * heights_agl_m[t_us]
            row = 2 * nobs + nb + m
            res[row] = (heights_agl_m[t_us] - (sol[3 + nb + 3 * j + 2] - c[2])) / sig_h
            jac[row, 3 + nb + 3 * j + 2] = 1.0 / sig_h
            jac[row, 2] = -tk / sig_h
            if nb:
                jac[row, 3:6] = a_int[2] / sig_h
            jac[row, ng:ng + 2] = -(0.5 * tk * tk) * g_jac[2] / sig_h
        try:
            dx = np.linalg.lstsq(jac, res, rcond=None)[0]
        except np.linalg.LinAlgError:
            return ViInitResult(False, "SINGULAR", features=nfeat, observations=nobs)
        sol = sol + dx
        if np.linalg.norm(dx[0:3 + nb]) < 1e-6:
            break
    try:
        cov = np.linalg.inv(jac.T @ jac)
    except np.linalg.LinAlgError:
        return ViInitResult(False, "SINGULAR", features=nfeat, observations=nobs)
    # Model check: a covariance is only meaningful if the model explains the
    # pixels. AHRS tilt errors leak gravity that looks like a bias and can
    # shrink the scale while the formal sigma stays small.
    chi2_dof = float(res[:2 * nobs] @ res[:2 * nobs]) / max(1, 2 * nobs - n)
    v0 = sol[0:3]
    ba = sol[3:6] if nb else np.zeros(3)
    t_end, r_end, _, _ = states[frames[-1][0]]
    jv = np.zeros((3, n))
    jv[:, 0:3] = np.eye(3)
    if nb:
        jv[:, 3:6] = -pre.b_int
    jv[:, ng:ng + 2] = g_jac * t_end
    g_est = gravity(sol)
    v_end = v0 + g_est * t_end + pre.beta - pre.b_int @ ba
    cov_v_end = jv @ cov @ jv.T
    # Rotate the solution into a frame where gravity is exactly down again
    # (what the filter assumes): rotation taking g_est onto GRAVITY_NED.
    q_fix = _rotation_between(g_est, GRAVITY_NED)
    v_end = q_fix @ v_end
    cov_v_end = q_fix @ cov_v_end @ q_fix.T
    q_end = _quat_from_matrix(q_fix @ rotation_body_to_ned(pre.q))
    tilt_deg = tuple(float(np.degrees(x)) for x in sol[ng:ng + 2])
    sigma_v = float(np.sqrt(np.max(np.linalg.eigvalsh(cov[0:3, 0:3]))))
    sigma_ba = float(np.sqrt(np.max(np.linalg.eigvalsh(cov[3:6, 3:6])))) if nb else 0.0
    # Excitation: RMS of the world acceleration about its window mean. Without
    # a range constraint the scale comes from its changes only; on real flights
    # with ~0.1-0.3 m/s^2 the formal sigma was small while the scale was off by
    # 2-5x (MUN-FRL): the covariance cannot be trusted there.
    acc = np.asarray(pre.acc_world)
    excitation = float(np.sqrt(np.mean(np.sum((acc - acc.mean(axis=0)) ** 2, axis=1)))) if len(acc) else 0.0
    excited = heights_agl_m is not None or excitation >= min_excitation_m_s2
    ok = (sigma_v <= max_sigma_v and sigma_ba <= max_sigma_ba and np.isfinite(v_end).all()
          and chi2_dof <= max_chi2_per_dof and excited)
    status = ("INITIALIZED" if ok else "MODEL_INCONSISTENT" if chi2_dof > max_chi2_per_dof
              else "INSUFFICIENT_EXCITATION")
    return ViInitResult(ok, status, frames[-1][0], tuple(float(x) for x in v_end),
                        tuple(tuple(float(x) for x in row) for row in cov_v_end),
                        tuple(float(x) for x in ba), tuple(float(x) for x in q_end),
                        tuple(float(x) for x in v0), sigma_v, sigma_ba, nfeat, nobs,
                        _median_depth(sol, tracks, states, nb, p_bc), chi2_dof, tilt_deg, excitation)
