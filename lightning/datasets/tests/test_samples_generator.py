import numpy as np

from sklearn.utils.testing import assert_equal

from lightning.datasets.samples_generator import make_nn_regression


def test_make_nn_regression():
    X, y, w = make_nn_regression(n_samples=10, n_features=50, n_informative=5)
    assert_equal(X.shape[0], 10)
    assert_equal(X.shape[1], 50)
    assert_equal(y.shape[0], 10)
    assert_equal(w.shape[0], 50)
    assert_equal(np.sum(X.data != 0), 10 * 5)

    X, y, w = make_nn_regression(n_samples=10, n_features=50, n_informative=50)
    assert_equal(np.sum(X.data != 0), 10 * 50)
