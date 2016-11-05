import numpy as np
from lightning.impl import penalty


def test_tv1_prox():
    """
    Use the properties of strongly convex functions to test the implementation
    of the TV1D proximal operator. In particular, we use the following inequality
    applied to the proximal objective function: if f is mu-strongly convex then

          f(x) - f(x^*) >= ||x - x^*||^2 / (2 mu)

    where x^* is the optimum of f.
    """
    n_features = 10
    gamma = np.random.rand()
    pen = penalty.TotalVariation1DPenalty()

    for nrun in range(5):
        x = np.random.randn(1, n_features)
        x2 = pen.projection(x, gamma, 1)
        diff_obj = pen.regularization(x) - pen.regularization(x2)
        assert diff_obj >= ((x - x2) ** 2).sum() / (2 * gamma)


def test_tv2_prox():
    """
    similar test, but for 2D total variation penalty.
    """
    n_features = 36
    gamma = np.random.rand()
    pen = penalty.TotalVariation2DPenalty(6, 6)

    for nrun in range(5):
        x = np.random.randn(1, n_features)
        x2 = pen.projection(x, gamma, 1)
        diff_obj = pen.regularization(x) - pen.regularization(x2)
        assert diff_obj >= ((x - x2) ** 2).sum() / (2 * gamma)