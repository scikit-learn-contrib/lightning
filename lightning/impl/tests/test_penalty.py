import numpy as np

from six.moves import xrange

from lightning.impl.penalty import project_l1_ball, project_simplex


def project_simplex_bisection(v, z=1, tau=0.0001, max_iter=1000):
    lower = 0
    upper = np.max(v)
    current = np.inf

    for it in xrange(max_iter):
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


def test_proj_simplex():
    rng = np.random.RandomState(0)

    v = rng.rand(100)
    w = project_simplex(v, z=10)
    w2 = project_simplex_bisection(v, z=10, max_iter=100)
    np.testing.assert_array_almost_equal(w, w2, 3)

    v = rng.rand(3)
    w = project_simplex(v, z=1)
    w2 = project_simplex_bisection(v, z=1, max_iter=100)
    np.testing.assert_array_almost_equal(w, w2, 3)

    v = rng.rand(2)
    w = project_simplex(v, z=1)
    w2 = project_simplex_bisection(v, z=1, max_iter=100)
    np.testing.assert_array_almost_equal(w, w2, 3)


def test_proj_l1_ball():
    rng = np.random.RandomState(0)
    v = rng.randn(100)
    w = project_l1_ball(v, z=50)
    np.testing.assert_almost_equal(np.sum(np.abs(w)), 50)
