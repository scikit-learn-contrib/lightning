import numpy as np

from lightning.impl.datasets.samples_generator import make_nn_regression


def test_make_nn_regression():
    X, y, w = make_nn_regression(n_samples=10, n_features=50, n_informative=5)
    assert X.shape[0] == 10
    assert X.shape[1] == 50
    assert y.shape[0] == 10
    assert w.shape[0] == 50
    assert np.sum(X.data != 0) == 10 * 5

    X, y, w = make_nn_regression(n_samples=10, n_features=50, n_informative=50)
    assert np.sum(X.data != 0) == 10 * 50
