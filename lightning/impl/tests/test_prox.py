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
            prox_fast.prox_tv1d(x, 1.0)
        # check that the solution is flat
        np.testing.assert_allclose(x, x.mean() * np.ones(n_features))
