"""Тесно связанная VIO: MSCKF (Mourikis, Roumeliotis, 2007), офлайн-прототип.

В отличие от слабосвязанной схемы (VO дает скорость, РФК ее сливает),
здесь фильтр сам использует пиксельные наблюдения признаков в нескольких
кадрах. Состояние ошибки: 15 состояний ИИМ (как в
``InertialErrorStateFilter``: δp, δv, δθ правой ошибкой, δbg, δba) и
скользящее окно клонов поз ИИМ на моменты кадров (δp, δθ по 6 на клон).

Обновление по треку признака, завершенному или уходящему из окна:

1. триангуляция точки по текущим оценкам клонов (DLT + Гаусс-Ньютон);
2. невязки репроекции 2M и якобианы по клонам H_x и по точке H_f;
3. проекция на левое нуль-пространство H_f — точка исключается из
   состояния, остается ограничение только на позы;
4. χ² по каждому треку (95 %), затем одно совместное обновление.

Масштаб наблюдается через ускорения ИИМ, а не через внешнюю высоту:
дальномер и высота над землей НЕ используются. Курс по-прежнему
ненаблюдаем (как в любой VIO без внешнего курса).

Камера — проективная, кадры уже без дисторсии. Внешняя калибровка
камера–ИИМ и сдвиг времени считаются известными. Результат НЕ передается
в EKF2.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite

import numpy as np

from .error_state_inertial_filter import (
    BA, BG, P, TH, V, ImuNoiseDensity, continuous_error_jacobian,
    discrete_process_covariance, exp_transition, skew,
)
from .imu_preintegration import (
    GRAVITY_NED, ImuBias, ImuReading, _unit_quaternion, quat_exponential, quat_product,
    rotation_body_to_ned,
)

# χ²(0,95) для 1..100 степеней свободы (Уилсон–Хилферти, точность ~1 %).
def _chi2_95(dof: int) -> float:
    z = 1.6448536269514722
    k = float(dof)
    return k * (1.0 - 2.0 / (9.0 * k) + z * np.sqrt(2.0 / (9.0 * k))) ** 3


@dataclass(frozen=True)
class PinholeCamera:
    fx: float
    fy: float
    cx: float
    cy: float
    body_from_camera: tuple[tuple[float, float, float], ...]
    camera_position_body_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    pixel_sigma: float = 1.0

    def matrices(self) -> tuple[np.ndarray, np.ndarray]:
        r = np.asarray(self.body_from_camera, dtype=float)
        p = np.asarray(self.camera_position_body_m, dtype=float)
        if (r.shape != (3, 3) or p.shape != (3,) or not np.allclose(r.T @ r, np.eye(3), atol=1e-3)
                or not np.isfinite(p).all() or not all(isfinite(x) and x > 0 for x in (self.fx, self.fy))
                or not self.pixel_sigma > 0):
            raise ValueError("Некорректная модель или внешняя калибровка камеры")
        return r, p


@dataclass
class _Clone:
    t_us: int
    p: np.ndarray
    q: np.ndarray


@dataclass
class MsckfStats:
    frames: int = 0
    features_used: int = 0
    features_rejected_chi2: int = 0
    features_rejected_triangulation: int = 0
    updates: int = 0
    counts: dict = field(default_factory=dict)
    # γ/dof of accepted tracks (projected residual NIS per degree of freedom):
    # its level calibrates the pixel noise on a separate flight.
    nis_per_dof: list = field(default_factory=list)


class Msckf:
    def __init__(self, *, camera: PinholeCamera, noise: ImuNoiseDensity,
                 quaternion_body_to_ned: tuple[float, float, float, float],
                 initial_position_ned_m=(0.0, 0.0, 0.0), initial_velocity_ned_m_s=(0.0, 0.0, 0.0),
                 initial_bias: ImuBias = ImuBias(),
                 initial_error_sigma: tuple[float, ...] = (
                     0.01, 0.01, 0.01, 1.0, 1.0, 1.0, 0.02, 0.02, 0.1,
                     0.01, 0.01, 0.01, 0.2, 0.2, 0.2),
                 max_clones: int = 10, min_track_length: int = 3,
                 min_parallax_rad: float = 0.003, max_imu_dt_s: float = 0.05) -> None:
        if not noise.valid() or max_clones < 3 or min_track_length < 3:
            raise ValueError("Недопустимые параметры MSCKF")
        self.r_bc, self.p_bc = camera.matrices()
        self.cam = camera
        self.noise = noise
        self.p = np.asarray(initial_position_ned_m, dtype=float)
        self.v = np.asarray(initial_velocity_ned_m_s, dtype=float)
        self.q = _unit_quaternion(quaternion_body_to_ned)
        self.bg = np.asarray(initial_bias.gyro_body_rad_s, dtype=float)
        self.ba = np.asarray(initial_bias.accel_body_m_s2, dtype=float)
        sig = np.asarray(initial_error_sigma, dtype=float)
        if sig.shape != (15,) or np.min(sig) <= 0:
            raise ValueError("Нужны 15 положительных начальных СКО")
        self.cov = np.diag(sig ** 2)
        self.clones: list[_Clone] = []
        self.tracks: dict[int, list[tuple[int, float, float]]] = {}  # id -> [(t_us, u, v)]
        self.max_clones = max_clones
        self.min_track = min_track_length
        self.min_parallax = min_parallax_rad
        self.max_imu_dt = max_imu_dt_s
        self.last_imu: ImuReading | None = None
        self.stats = MsckfStats()

    # ------------------------------------------------------------- IMU
    def predict(self, reading: ImuReading) -> None:
        if self.last_imu is None:
            self.last_imu = reading
            return
        dt = (reading.timestamp_us - self.last_imu.timestamp_us) * 1e-6
        if not 0 < dt <= self.max_imu_dt:
            raise ValueError("Разрыв или нарушение порядка ИИМ")
        omega = 0.5 * (np.asarray(reading.gyro_body_rad_s) + np.asarray(self.last_imu.gyro_body_rad_s)) - self.bg
        f_b = 0.5 * (np.asarray(reading.specific_force_body_m_s2)
                     + np.asarray(self.last_imu.specific_force_body_m_s2)) - self.ba
        qmid = _unit_quaternion(quat_product(self.q, quat_exponential(0.5 * dt * omega)))
        rmid = rotation_body_to_ned(qmid)
        a_nav = rmid @ f_b + GRAVITY_NED
        f = continuous_error_jacobian(rmid, omega, f_b)
        phi = exp_transition(f, dt)
        qd = discrete_process_covariance(f, rmid, dt, self.noise)
        self.p = self.p + dt * self.v + 0.5 * dt * dt * a_nav
        self.v = self.v + dt * a_nav
        self.q = _unit_quaternion(quat_product(self.q, quat_exponential(dt * omega)))
        n = self.cov.shape[0]
        new = np.empty_like(self.cov)
        new[:15, :15] = phi @ self.cov[:15, :15] @ phi.T + qd
        if n > 15:
            new[:15, 15:] = phi @ self.cov[:15, 15:]
            new[15:, :15] = new[:15, 15:].T
            new[15:, 15:] = self.cov[15:, 15:]
        self.cov = 0.5 * (new + new.T)
        self.last_imu = reading

    # ------------------------------------------------------------ frames
    def _augment(self, t_us: int) -> None:
        n = self.cov.shape[0]
        j = np.zeros((6, n))
        j[0:3, P] = np.eye(3)
        j[3:6, TH] = np.eye(3)
        top = np.hstack([self.cov, self.cov @ j.T])
        bottom = np.hstack([j @ self.cov, j @ self.cov @ j.T])
        self.cov = np.vstack([top, bottom])
        self.clones.append(_Clone(t_us, self.p.copy(), self.q.copy()))

    def _drop_clone(self, index: int) -> None:
        a = 15 + 6 * index
        keep = np.r_[0:a, a + 6:self.cov.shape[0]]
        self.cov = self.cov[np.ix_(keep, keep)]
        t = self.clones.pop(index).t_us
        for obs in self.tracks.values():
            obs[:] = [o for o in obs if o[0] != t]

    def process_frame(self, t_us: int, features: dict[int, tuple[float, float]]) -> dict:
        """Кадр на момент последнего отсчета ИИМ (t_us == last_imu.timestamp_us)."""
        if self.last_imu is None or t_us != self.last_imu.timestamp_us:
            raise ValueError("Кадр должен совпадать по времени с последним отсчетом ИИМ")
        self.stats.frames += 1
        self._augment(t_us)
        for fid, (u, v) in features.items():
            self.tracks.setdefault(fid, []).append((t_us, float(u), float(v)))
        lost = [fid for fid in self.tracks if fid not in features]
        to_use = {fid: self.tracks.pop(fid) for fid in lost}
        if len(self.clones) > self.max_clones:
            # Oldest clone leaves the window: use every track still seeing it.
            t_old = self.clones[0].t_us
            for fid, obs in self.tracks.items():
                if obs and obs[0][0] == t_old and len(obs) >= self.min_track:
                    to_use[fid] = list(obs)
            used = self._update([obs for obs in to_use.values() if len(obs) >= self.min_track])
            for fid in list(to_use):
                # Each observation is used once: a track used here starts
                # afresh from the next frame (no double counting).
                self.tracks.pop(fid, None)
            self._drop_clone(0)
        else:
            used = self._update([obs for obs in to_use.values() if len(obs) >= self.min_track])
        return {"used_features": used, "clones": len(self.clones)}

    # ---------------------------------------------------------- geometry
    def _camera_pose(self, clone: _Clone) -> tuple[np.ndarray, np.ndarray]:
        r_wb = rotation_body_to_ned(clone.q)
        return r_wb @ self.r_bc, clone.p + r_wb @ self.p_bc

    def _triangulate(self, obs, poses) -> np.ndarray | None:
        """Inverse-depth Gauss-Newton anchored at the first observation.

        Robust for distant points with little parallax (a 30-m nadir view
        over a 1-m baseline): depth is a well-conditioned parameter only as
        inverse depth. Returns the world point or None.
        """
        r_a, p_a = poses[0]
        rays = []
        for (_, u, v), (r_wc, _) in zip(obs, poses):
            ray = r_wc @ np.array([(u - self.cam.cx) / self.cam.fx, (v - self.cam.cy) / self.cam.fy, 1.0])
            rays.append(ray / np.linalg.norm(ray))
        cos_min = min(float(a @ b) for a in rays for b in rays)
        if np.arccos(np.clip(cos_min, -1.0, 1.0)) < self.min_parallax:
            return None
        # Initial inverse depth: linear least squares over all observations
        # with the anchor bearing fixed (rays are nearly parallel, so a
        # two-ray midpoint is ill-conditioned and often negative).
        bearing0 = r_a.T @ rays[0]
        alpha, beta = bearing0[0] / bearing0[2], bearing0[1] / bearing0[2]
        num = den = 0.0
        for (_, u, v), (r_wc, p_wc) in zip(obs[1:], poses[1:]):
            r_ca = r_wc.T @ r_a
            t = r_wc.T @ (p_a - p_wc)
            a = r_ca @ np.array([alpha, beta, 1.0])
            un, vn = (u - self.cam.cx) / self.cam.fx, (v - self.cam.cy) / self.cam.fy
            # (un*h_z - h_x) = 0 and (vn*h_z - h_y) = 0 with h = a + rho t
            ca = np.array([un * a[2] - a[0], vn * a[2] - a[1]])
            cb = np.array([un * t[2] - t[0], vn * t[2] - t[1]])
            num -= float(ca @ cb)
            den += float(cb @ cb)
        if den <= 0:
            return None
        x = np.array([alpha, beta, num / den])
        if not x[2] > 1e-4:
            return None
        for _ in range(10):
            jt_j = np.zeros((3, 3))
            jt_r = np.zeros(3)
            for (_, u, v), (r_wc, p_wc) in zip(obs, poses):
                # h = R_wc^T (p_a + R_a [a, b, 1]/rho - p_wc), scaled by rho:
                r_ca = r_wc.T @ r_a
                t = r_wc.T @ (p_a - p_wc)
                hv = r_ca @ np.array([x[0], x[1], 1.0]) + x[2] * t
                if hv[2] <= 1e-6:
                    return None
                z = np.array([self.cam.fx * hv[0] / hv[2] + self.cam.cx, self.cam.fy * hv[1] / hv[2] + self.cam.cy])
                jh = np.array([[self.cam.fx / hv[2], 0.0, -self.cam.fx * hv[0] / hv[2] ** 2],
                               [0.0, self.cam.fy / hv[2], -self.cam.fy * hv[1] / hv[2] ** 2]])
                j = jh @ np.column_stack([r_ca[:, 0], r_ca[:, 1], t])
                r = np.array([u, v]) - z
                jt_j += j.T @ j
                jt_r += j.T @ r
            try:
                dx = np.linalg.solve(jt_j + 1e-9 * np.eye(3), jt_r)
            except np.linalg.LinAlgError:
                return None
            x = x + dx
            if np.linalg.norm(dx) < 1e-7:
                break
        if not x[2] > 1e-4:  # behind the camera or beyond 10 km
            return None
        return p_a + r_a @ np.array([x[0], x[1], 1.0]) / x[2]

    def _project(self, pc: np.ndarray) -> np.ndarray:
        return np.array([self.cam.fx * pc[0] / pc[2] + self.cam.cx, self.cam.fy * pc[1] / pc[2] + self.cam.cy])

    def _proj_jacobian(self, pc: np.ndarray) -> np.ndarray:
        x, y, z = pc
        return np.array([[self.cam.fx / z, 0.0, -self.cam.fx * x / z ** 2],
                         [0.0, self.cam.fy / z, -self.cam.fy * y / z ** 2]])

    def _feature_system(self, obs) -> tuple[np.ndarray, np.ndarray] | None:
        index = {c.t_us: i for i, c in enumerate(self.clones)}
        obs = [o for o in obs if o[0] in index]
        if len(obs) < self.min_track:
            return None
        clones = [self.clones[index[o[0]]] for o in obs]
        poses = [self._camera_pose(c) for c in clones]
        pf = self._triangulate(obs, poses)
        if pf is None:
            self.stats.features_rejected_triangulation += 1
            return None
        n = self.cov.shape[0]
        m = len(obs)
        hx = np.zeros((2 * m, n))
        hf = np.zeros((2 * m, 3))
        r = np.zeros(2 * m)
        for k, ((_, u, v), clone) in enumerate(zip(obs, clones)):
            r_wb = rotation_body_to_ned(clone.q)
            y = r_wb.T @ (pf - clone.p)              # point in body
            pc = self.r_bc.T @ (y - self.p_bc)        # point in camera
            if pc[2] <= 0.1:
                self.stats.features_rejected_triangulation += 1
                return None
            jp = self._proj_jacobian(pc) @ self.r_bc.T
            col = 15 + 6 * index[clone.t_us]
            hx[2 * k:2 * k + 2, col:col + 3] = jp @ (-r_wb.T)   # δp
            hx[2 * k:2 * k + 2, col + 3:col + 6] = jp @ skew(y)  # δθ (right error)
            hf[2 * k:2 * k + 2] = jp @ r_wb.T
            r[2 * k:2 * k + 2] = np.array([u, v]) - self._project(pc)
        if float(np.sqrt(np.mean(r ** 2))) > 20.0 * self.cam.pixel_sigma:  # gross failure only
            self.stats.features_rejected_triangulation += 1
            return None
        # Left null space of H_f.
        qmat, _ = np.linalg.qr(hf, mode="complete")
        a = qmat[:, 3:]
        return a.T @ hx, a.T @ r

    def _update(self, tracks) -> int:
        hs, rs = [], []
        sigma2 = self.cam.pixel_sigma ** 2
        for obs in tracks:
            sys_ = self._feature_system(obs)
            if sys_ is None:
                continue
            h, r = sys_
            s = h @ self.cov @ h.T + sigma2 * np.eye(len(r))
            try:
                gamma = float(r @ np.linalg.solve(s, r))
            except np.linalg.LinAlgError:
                continue
            if gamma > _chi2_95(len(r)):
                self.stats.features_rejected_chi2 += 1
                continue
            hs.append(h)
            rs.append(r)
            self.stats.nis_per_dof.append(gamma / len(r))
        if not hs:
            return 0
        h = np.vstack(hs)
        r = np.concatenate(rs)
        # QR compression when there are more rows than states.
        if h.shape[0] > h.shape[1]:
            qmat, rmat = np.linalg.qr(h, mode="reduced")
            h, r = rmat, qmat.T @ r
        s = h @ self.cov @ h.T + sigma2 * np.eye(len(r))
        k = np.linalg.solve(s, h @ self.cov).T
        dx = k @ r
        a = np.eye(self.cov.shape[0]) - k @ h
        cov = a @ self.cov @ a.T + sigma2 * (k @ k.T)
        self.cov = 0.5 * (cov + cov.T)
        self.p = self.p + dx[P]
        self.v = self.v + dx[V]
        self.q = _unit_quaternion(quat_product(self.q, quat_exponential(dx[TH])))
        self.bg = self.bg + dx[BG]
        self.ba = self.ba + dx[BA]
        for i, c in enumerate(self.clones):
            d = dx[15 + 6 * i:21 + 6 * i]
            c.p = c.p + d[:3]
            c.q = _unit_quaternion(quat_product(c.q, quat_exponential(d[3:])))
        self.stats.features_used += len(hs)
        self.stats.updates += 1
        return len(hs)
