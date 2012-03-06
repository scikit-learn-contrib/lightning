import numpy as np
import scipy.sparse as sp

from numpy.testing import assert_array_equal, assert_array_almost_equal, \
                          assert_almost_equal
from nose.tools import assert_raises, assert_true, assert_equal

from sklearn.datasets.samples_generator import make_classification
from sklearn.metrics.pairwise import pairwise_kernels

from lightning.kernel_fast import RbfKernel
from lightning.kernel_fast import LinearKernel
from lightning.kernel_fast import PolynomialKernel
from lightning.kernel_fast import PrecomputedKernel


X, _ = make_classification(n_samples=20, n_features=10,
                           n_informative=5,
                           n_classes=2, random_state=0)

def _test_equal(K, kernel):
    n_samples = K.shape[0]
    for i in xrange(n_samples):
        for j in xrange(n_samples):
            assert_almost_equal(K[i,j], kernel.compute(X, i, X, j))


def test_linear_kernel():
    n_samples = X.shape[0]
    K = pairwise_kernels(X, metric="linear")
    kernel = LinearKernel()
    _test_equal(K, kernel)


def test_polynomial_kernel():
    n_samples = X.shape[0]
    K = pairwise_kernels(X, metric="polynomial", degree=3, coef0=1.0, gamma=0.1)
    kernel = PolynomialKernel(degree=3, coef0=1.0, gamma=0.1)
    _test_equal(K, kernel)


def test_rbf_kernel():
    n_samples = X.shape[0]
    K = pairwise_kernels(X, metric="rbf", gamma=0.1)
    kernel = RbfKernel(gamma=0.1)
    _test_equal(K, kernel)


def test_precomputed_kernel():
    K = np.dot(X, X.T)
    kernel = PrecomputedKernel(K)
    _test_equal(K, kernel)
