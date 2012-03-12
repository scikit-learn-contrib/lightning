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
from lightning.kernel_fast import KernelCache


X, _ = make_classification(n_samples=20, n_features=10,
                           n_informative=5,
                           n_classes=2, random_state=0)
X2 = X ** 2

capacity = 4 * 20 * 8 # 4 columns of 20 doubles

def _test_equal(K, kernel, X2):
    n_samples = K.shape[0]

    # test compute()
    for i in xrange(n_samples):
        for j in xrange(n_samples):
            assert_almost_equal(K[i,j], kernel.compute(X2, i, X2, j))

def test_linear_kernel():
    n_samples = X.shape[0]
    K = pairwise_kernels(X, metric="linear")
    kernel = LinearKernel()
    _test_equal(K, kernel, X)


def test_polynomial_kernel():
    n_samples = X.shape[0]
    K = pairwise_kernels(X, metric="polynomial", degree=3, coef0=1.0, gamma=0.1)
    kernel = PolynomialKernel(degree=3, coef0=1.0, gamma=0.1)
    _test_equal(K, kernel, X)


def test_rbf_kernel():
    n_samples = X.shape[0]
    K = pairwise_kernels(X, metric="rbf", gamma=0.1)
    kernel = RbfKernel(gamma=0.1)
    _test_equal(K, kernel, X)


def test_precomputed_kernel():
    K = np.dot(X, X.T)
    kernel = PrecomputedKernel()
    _test_equal(K, kernel, K)


def test_kernel_cache_compute():
    kernel = RbfKernel(gamma=0.1)
    kcache = KernelCache(kernel, 20, capacity)
    assert_equal(kcache.compute(X, 0, X, 0), 1.0)


def test_kernel_cache_add_remove():
    kernel = RbfKernel(gamma=0.1)
    kcache = KernelCache(kernel, 20, capacity)

    for i in xrange(3):
        kcache.add_sv(i)

    assert_equal(kcache.n_sv(), 3)

    kcache.remove_sv(1)
    assert_equal(kcache.n_sv(), 2)


def test_kernel_cache_column():
    K = pairwise_kernels(X, metric="rbf", gamma=0.1)
    kernel = RbfKernel(gamma=0.1)
    kcache = KernelCache(kernel, 20, capacity)
    out = np.zeros(20, dtype=np.float64)

    # Compute a first column.
    kcache.compute_column(X, X, 12, out)
    assert_array_almost_equal(K[:, 12], out)
    assert_equal(kcache.get_size(), 160)

    # Check that the works.
    kcache.compute_column(X2, X2, 12, out)
    assert_array_almost_equal(K[:, 12], out)
    assert_equal(kcache.get_size(), 160)

    # Compute more columns.
    kcache.compute_column(X, X, 13, out)
    assert_array_almost_equal(K[:, 13], out)
    assert_equal(kcache.get_size(), 320)

    kcache.compute_column(X, X, 14, out)
    assert_array_almost_equal(K[:, 14], out)
    assert_equal(kcache.get_size(), 480)

    kcache.compute_column(X, X, 15, out)
    assert_array_almost_equal(K[:, 15], out)
    assert_equal(kcache.get_size(), 640)

    # Maximum size reached.
    kcache.compute_column(X, X, 16, out)
    assert_array_almost_equal(K[:, 16], out)
    assert_equal(kcache.get_size(), 480)

    # Check that cache works.
    kcache.compute_column(X2, X2, 16, out)
    assert_array_almost_equal(K[:, 16], out)
    assert_equal(kcache.get_size(), 480)


def test_kernel_cache_column_sv():
    K = pairwise_kernels(X, metric="rbf", gamma=0.1)
    kernel = RbfKernel(gamma=0.1)
    kcache = KernelCache(kernel, 20, capacity)
    out = np.zeros(20, dtype=np.float64)

    size = 0

    # Add 3 SVs.
    kcache.add_sv(6)
    kcache.add_sv(12)
    kcache.add_sv(3)
    size += 3 * 8

    # Compute values.
    kcache.compute_column_sv(X, X, 7, out)
    assert_almost_equal(K[6, 7], out[6])
    assert_almost_equal(K[12, 7], out[12])
    assert_almost_equal(K[3, 7], out[3])
    assert_equal(size, kcache.get_size())

    # Check that the works.
    kcache.compute_column_sv(X2, X2, 7, out)
    assert_almost_equal(K[6, 7], out[6])
    assert_almost_equal(K[12, 7], out[12])
    assert_almost_equal(K[3, 7], out[3])
    assert_equal(size, kcache.get_size())

    # Add one more SV.
    kcache.add_sv(17)
    size += 8

    # Compute values.
    kcache.compute_column_sv(X, X, 7, out)
    assert_almost_equal(K[6, 7], out[6])
    assert_almost_equal(K[12, 7], out[12])
    assert_almost_equal(K[3, 7], out[3])
    assert_almost_equal(K[17, 7], out[17])
    assert_equal(size, kcache.get_size())

    # Check that the works.
    kcache.compute_column_sv(X2, X2, 7, out)
    assert_almost_equal(K[6, 7], out[6])
    assert_almost_equal(K[12, 7], out[12])
    assert_almost_equal(K[3, 7], out[3])
    assert_almost_equal(K[17, 7], out[17])
    assert_equal(size, kcache.get_size())

    # Compute the entire same column.
    kcache.compute_column(X, X, 7, out)
    size = 20 * 8
    assert_array_almost_equal(K[:, 7], out)
    assert_equal(size, kcache.get_size())

    # Compute an entire new column.
    kcache.compute_column(X, X, 8, out)
    size += 20 * 8
    assert_array_almost_equal(K[:, 8], out)
    assert_equal(size, kcache.get_size())

    # Retrieve the SV only.
    out *= 0
    kcache.compute_column_sv(X, X, 8, out)
    assert_almost_equal(K[6, 8], out[6])
    assert_almost_equal(K[12, 8], out[12])
    assert_almost_equal(K[3, 8], out[3])
    assert_almost_equal(K[17, 8], out[17])
    assert_equal(size, kcache.get_size())

    # Remove SV.
    out *= 0
    kcache.remove_sv(12)
    kcache.compute_column_sv(X, X, 8, out)
    assert_almost_equal(K[6, 8], out[6])
    assert_almost_equal(0, out[12])
    assert_almost_equal(K[3, 8], out[3])
    assert_almost_equal(K[17, 8], out[17])
    assert_equal(size, kcache.get_size())

    for i in (12, 6, 3, 17):
        kcache.remove_sv(i)

    # Add back some new SV.
    kcache.add_sv(15)
    out *= 0
    kcache.compute_column_sv(X, X, 8, out)
    assert_almost_equal(0, out[6])
    assert_almost_equal(0, out[12])
    assert_almost_equal(K[15, 8], out[15])
    assert_equal(size, kcache.get_size())

    # Remove non-existing column.
    kcache.remove_column(19)
    assert_equal(size, kcache.get_size())

    # Remove existing column.
    kcache.remove_column(8)
    size -= 20 * 8
    assert_equal(size, kcache.get_size())

