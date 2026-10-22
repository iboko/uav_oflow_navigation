"""Wind disturbance for Gazebo SITL: drag force on the vehicle model.

The PX4 Gazebo models have no wind (no ``enable_wind``, no WindEffects
system), so wind is applied as a persistent link wrench on the model
(``gz-sim-apply-link-wrench-system``, topic
``/world/<world>/wrench/persistent``): the drag of a steady wind
F = k |w| w, k = 0.5 rho CdA, plus gusts. Gusts are a first-order
(exponentially correlated) random process per horizontal axis, seeded.

Every message on the persistent topic ADDS a wrench to the ones already
applied (a first version republishing the full force at 10 Hz built up
~18 N in a second and flipped the vehicle). Only the change of the force
is published, so the applied sum equals the current force; ``stop``
clears them.

Approximation: the drag depends on the wind only, not on the vehicle's
airspeed (no change when it flies with or against the wind); enough to
make the vehicle tilt and work against a disturbance, not a wind model.
"""
from __future__ import annotations

import math
import threading
import time

import numpy as np

AIR_DENSITY = 1.2
X500_CDA_M2 = 0.12   # assumed drag area of the x500 frame (not measured)


def drag_force_ne(wind_ne_m_s, cda_m2: float = X500_CDA_M2) -> np.ndarray:
    w = np.asarray(wind_ne_m_s, dtype=float)
    return 0.5 * AIR_DENSITY * cda_m2 * np.linalg.norm(w) * w


class GustProcess:
    """First-order gust speed per axis: dg = -g/tau dt + sigma sqrt(2/tau) dW."""

    def __init__(self, sigma_m_s: float, tau_s: float = 2.0, seed: int = 0) -> None:
        if sigma_m_s < 0 or tau_s <= 0:
            raise ValueError("gust sigma must be >= 0 and tau > 0")
        self.sigma, self.tau = sigma_m_s, tau_s
        self.rng = np.random.default_rng(seed)
        self.g = np.zeros(2)

    def step(self, dt_s: float) -> np.ndarray:
        if dt_s <= 0:
            return self.g.copy()
        a = math.exp(-dt_s / self.tau)
        self.g = a * self.g + self.sigma * math.sqrt(1 - a * a) * self.rng.standard_normal(2)
        return self.g.copy()


class WindDisturbance:
    """Publishes the wind wrench on a model at a fixed rate (sim-time gusts)."""

    def __init__(self, world: str, model: str, wind_ne_m_s, gust_sigma_m_s: float = 0.0,
                 gust_tau_s: float = 2.0, seed: int = 0, rate_hz: float = 10.0, log_path=None,
                 clock=None) -> None:
        """``clock``: callable returning simulation time (s); gusts evolve in it."""
        self.world, self.model = world, model
        self.wind = np.asarray(wind_ne_m_s, dtype=float)
        self.gusts = GustProcess(gust_sigma_m_s, gust_tau_s, seed)
        self.period = 1.0 / rate_hz
        self.clock = clock or time.time
        self.log = open(log_path, "w") if log_path else None
        if self.log:
            self.log.write("sim_s,wind_n,wind_e,force_n,force_e\n")
        self._applied = np.zeros(2)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _publish(self, force_ne: np.ndarray) -> None:
        from gz.msgs10.entity_wrench_pb2 import EntityWrench
        from gz.msgs10.entity_pb2 import Entity
        msg = EntityWrench()
        msg.entity.name = self.model
        msg.entity.type = Entity.MODEL
        # Gazebo world frame is ENU: x east, y north.
        msg.wrench.force.x = float(force_ne[1])
        msg.wrench.force.y = float(force_ne[0])
        self._pub.publish(msg)

    def start(self) -> None:
        from gz.msgs10.entity_wrench_pb2 import EntityWrench
        from gz.transport13 import Node
        self._node = Node()
        self._pub = self._node.advertise(f"/world/{self.world}/wrench/persistent", EntityWrench)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def update(self, now: float, dt: float) -> np.ndarray:
        """One step: publish the change of the drag force; returns the force."""
        w = self.wind + self.gusts.step(dt)
        f = drag_force_ne(w)
        self._publish(f - self._applied)
        self._applied = f
        if self.log:
            self.log.write(f"{now:.2f},{w[0]:.3f},{w[1]:.3f},{f[0]:.4f},{f[1]:.4f}\n")
            self.log.flush()
        return f

    def _run(self) -> None:
        last = self.clock()
        while not self._stop.is_set():
            now = self.clock()
            self.update(now, max(now - last, 0.0))
            last = now
            self._stop.wait(self.period)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        try:
            from gz.msgs10.entity_pb2 import Entity
            from gz.transport13 import Node
            pub = self._node.advertise(f"/world/{self.world}/wrench/clear", Entity)
            e = Entity()
            e.name, e.type = self.model, Entity.MODEL
            pub.publish(e)
        except Exception:
            pass
        if self.log:
            self.log.close()
