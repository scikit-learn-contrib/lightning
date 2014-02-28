import numpy as np

from sklearn.datasets import make_classification
from sklearn.linear_model import Ridge

from sklearn.utils.testing import assert_array_almost_equal

from lightning.regression import KernelRidge

X, y = make_classification(n_classes=2, random_state=0)
Y = np.array([y, y]).T


def test_kernel_ridge():
    pred = Ridge(alpha=1, fit_intercept=False).fit(X, y).predict(X)
    pred2 = KernelRidge(kernel="linear", alpha=1).fit(X, y).predict(X)
    assert_array_almost_equal(pred, pred2)


def test_kernel_ridge_precomputed():
    K = np.dot(X, X.T)
    pred = KernelRidge(kernel="linear").fit(X, y).predict(X)
    pred2 = KernelRidge(kernel="precomputed").fit(K, y).predict(K)
    assert_array_almost_equal(pred, pred2)


def test_kernel_ridge_multi_output():
    pred = Ridge(alpha=1, fit_intercept=False).fit(X, Y).predict(X)
    pred2 = KernelRidge(kernel="linear", alpha=1).fit(X, Y).predict(X)
    assert_array_almost_equal(pred, pred2)
