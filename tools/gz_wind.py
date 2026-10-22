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
clears them. Gazebo keeps every message in a list it reapplies each
physics step, so the list is bounded: changes below 1 mN are not sent,
and every ``consolidate_s`` of sim time the list is cleared and replaced
by one wrench of the full force (a clear removes all wrenches queued
before it; the full force goes out after a short wait).

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
                 clock=None, consolidate_s: float = 10.0) -> None:
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
        self.consolidate_s = consolidate_s
        self._last_consolidated: float | None = None
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

    def _clear(self) -> None:
        from gz.msgs10.entity_pb2 import Entity
        e = Entity()
        e.name, e.type = self.model, Entity.MODEL
        self._clear_pub.publish(e)

    def start(self) -> None:
        from gz.msgs10.entity_pb2 import Entity
        from gz.msgs10.entity_wrench_pb2 import EntityWrench
        from gz.transport13 import Node
        self._node = Node()
        self._pub = self._node.advertise(f"/world/{self.world}/wrench/persistent", EntityWrench)
        self._clear_pub = self._node.advertise(f"/world/{self.world}/wrench/clear", Entity)
        # A message published before Gazebo has subscribed is lost; with
        # change-only publishing that would drop the steady force for good
        # (second wind flight: tilt answered only the gusts).
        t_end = time.time() + 20.0
        while not (self._pub.has_connections() and self._clear_pub.has_connections()):
            if time.time() > t_end:
                raise RuntimeError("no subscriber on the wrench topics (ApplyLinkWrench system loaded?)")
            time.sleep(0.05)
        time.sleep(0.5)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def update(self, now: float, dt: float) -> np.ndarray:
        """One step: publish the change of the drag force; returns the force."""
        w = self.wind + self.gusts.step(dt)
        f = drag_force_ne(w)
        if self._last_consolidated is None or now - self._last_consolidated >= self.consolidate_s:
            if self._last_consolidated is not None:
                self._clear()
                time.sleep(0.1)       # the clear must reach Gazebo before the new wrench
            self._publish(f)
            self._applied = f
            self._last_consolidated = now
        elif np.max(np.abs(f - self._applied)) >= 1e-3:
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
            self._clear()
            time.sleep(0.2)           # keep the node alive until the clear is delivered
        except Exception:
            pass
        if self.log:
            self.log.close()
