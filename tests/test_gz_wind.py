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
