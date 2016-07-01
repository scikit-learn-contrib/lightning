import numpy as np
from lightning.impl import prox_fast

def test_tv1_denoise():
    # test the prox of TV1D
    # since its not trivial to check the KKT conditions
    # we check that the proximal point algorithm converges
    # to a solution to the TV minimization
    n_iter = 100
    n_features = 100

    # repeat the test 10 times
    for nrun in range(10):
        x = np.random.randn(n_features)
        for _ in range(n_iter):
            x = prox_fast.prox_tv1d(x, 1.0)
        # check that the solution is flat
        np.testing.assert_allclose(x, x.mean() * np.ones(n_features))


def test_noncontiguous():

    a0 = np.random.randn(10)[::2]
    # argument is not contiguous
    p1 = prox_fast.prox_tv1d(a0, 1.0)
    # argument is contiguous
    p2 = prox_fast.prox_tv1d(a0.copy(), 1.0)
    np.testing.assert_array_equal(p1, p2)
