import numpy as np
import pytest

from lightning.impl.penalty import project_l1_ball, project_simplex


def project_simplex_bisection(v, z=1, tau=0.0001, max_iter=1000):
    lower = 0
    upper = np.max(v)
    current = np.inf

    for it in range(max_iter):
        if np.abs(current) / z < tau and current < 0:
            break

        theta = (upper + lower) / 2.0
        w = np.maximum(v - theta, 0)
        current = np.sum(w) - z
        if current <= 0:
            upper = theta
        else:
            lower = theta
    return w


@pytest.mark.parametrize("rng, z", [(100, 10),
                                    (3, 1),
                                    (2, 1)])
def test_proj_simplex(rng, z):
    rng = np.random.RandomState(0)

    v = rng.rand(rng)
    w = project_simplex(v, z=z)
    w2 = project_simplex_bisection(v, z=z, max_iter=100)
    np.testing.assert_array_almost_equal(w, w2, 3)


def test_proj_l1_ball():
    rng = np.random.RandomState(0)
    v = rng.randn(100)
    w = project_l1_ball(v, z=50)
    np.testing.assert_almost_equal(np.sum(np.abs(w)), 50)
