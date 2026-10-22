import numpy as np
import pytest

from tools.gz_wind import GustProcess, drag_force_ne


def test_drag_force_direction_and_size():
    f = drag_force_ne((5.0, 0.0), cda_m2=0.12)
    assert f == pytest.approx([0.5 * 1.2 * 0.12 * 25.0, 0.0])
    f = drag_force_ne((0.0, -3.0), cda_m2=0.1)
    assert f[0] == 0.0 and f[1] == pytest.approx(-0.5 * 1.2 * 0.1 * 9.0)


def test_gust_process_has_the_set_sigma_and_correlation():
    g = GustProcess(1.5, tau_s=2.0, seed=3)
    x = np.array([g.step(0.1) for _ in range(40000)])
    assert np.std(x[2000:], axis=0) == pytest.approx([1.5, 1.5], rel=0.1)
    lag = 20  # 2 s = tau: correlation exp(-1)
    c = np.corrcoef(x[2000:-lag, 0], x[2000 + lag:, 0])[0, 1]
    assert c == pytest.approx(np.exp(-1), abs=0.08)
    assert np.array_equal(g.step(0.0), g.g)   # no time, no change
    with pytest.raises(ValueError):
        GustProcess(-1.0)


def test_only_force_changes_are_published_and_the_list_stays_bounded(monkeypatch):
    monkeypatch.setattr("tools.gz_wind.time.sleep", lambda s: None)
    # Gazebo adds every persistent wrench message: the published sum must
    # equal the current drag force, not grow with the number of updates.
    from tools.gz_wind import WindDisturbance
    # Every consolidate_s the list is cleared and one full wrench sent.
    w = WindDisturbance("w", "m", (0.0, 5.0), gust_sigma_m_s=1.0, seed=2, consolidate_s=2.0)
    stored = []                          # Gazebo's persistent-wrench list
    w._publish = lambda f: stored.append(np.array(f))
    w._clear = lambda: stored.clear()
    longest = 0
    for k in range(200):
        f = w.update(k * 0.1, 0.1)
        assert np.sum(stored, axis=0) == pytest.approx(f, abs=1e-12)
        longest = max(longest, len(stored))
    assert np.linalg.norm(f) < 5.0      # ~1.8 N mean, not 200 x 1.8 N
    assert longest <= 21                # bounded by rate x consolidate_s
