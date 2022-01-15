import numpy as np
import pytest

from sklearn.datasets import load_diabetes

from lightning.ranking import PRank
from lightning.ranking import KernelPRank


@pytest.fixture(scope="module")
def train_data():
    bunch = load_diabetes()
    X, y = bunch.data, bunch.target
    y = np.round(y, decimals=-2)
    return X, y


def test_prank(train_data):
    X, y = train_data
    est = PRank(n_iter=10, shuffle=False, random_state=0)
    est.fit(X, y)
    np.testing.assert_almost_equal(est.score(X, y), 41.86, 2)

    est = PRank(n_iter=10, shuffle=True, random_state=0)
    est.fit(X, y)
    np.testing.assert_almost_equal(est.score(X, y), 71.04, 2)


def test_prank_linear_kernel(train_data):
    X, y = train_data
    est = KernelPRank(kernel="linear", n_iter=10, shuffle=False,
                      random_state=0)
    est.fit(X, y)
    np.testing.assert_almost_equal(est.score(X, y), 41.86, 2)


def test_prank_rbf_kernel(train_data):
    X, y = train_data
    est = KernelPRank(kernel="rbf", gamma=100, n_iter=10, shuffle=False,
                      random_state=0)
    est.fit(X, y)
    np.testing.assert_almost_equal(est.score(X, y), 15.84, 2)
