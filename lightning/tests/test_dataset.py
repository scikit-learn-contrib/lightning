import numpy as np
import scipy.sparse as sp

from numpy.testing import assert_array_equal, assert_array_almost_equal, \
                          assert_almost_equal
from nose.tools import assert_raises, assert_true, assert_equal, \
                       assert_not_equal

from sklearn.datasets.samples_generator import make_classification
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils import check_random_state

from lightning.dataset_fast import ContiguousDataset
from lightning.dataset_fast import FortranDataset
from lightning.dataset_fast import CSRDataset
from lightning.dataset_fast import CSCDataset
from lightning.dataset_fast import KernelDataset

# Create test datasets.
X, _ = make_classification(n_samples=20, n_features=100,
                           n_informative=5, n_classes=2, random_state=0)
X2, _ = make_classification(n_samples=10, n_features=100,
                            n_informative=5, n_classes=2, random_state=0)

# Sparsify datasets.
X[X < 0.3] = 0

capacity = 4 * 20 * 8 # 4 columns of 20 doubles

X_csr = sp.csr_matrix(X)
X_csc = sp.csc_matrix(X)

rs = check_random_state(0)


def test_contiguous_get_row():
    ind = np.arange(X.shape[1])
    ds = ContiguousDataset(X)
    for i in xrange(X.shape[0]):
        indices, data, n_nz = ds.get_row(i)
        assert_array_equal(indices, ind)
        assert_array_equal(data, X[i])
        assert_equal(n_nz, X.shape[1])


def test_csr_get_row():
    ds = CSRDataset(X_csr)
    for i in xrange(X.shape[0]):
        indices, data, n_nz = ds.get_row(i)
        for jj in xrange(n_nz):
            j = indices[jj]
            assert_equal(X[i, j], data[jj])


def test_fortran_get_column():
    ind = np.arange(X.shape[0])
    ds = FortranDataset(np.asfortranarray(X))
    for j in xrange(X.shape[1]):
        indices, data, n_nz = ds.get_column(j)
        assert_array_equal(indices, ind)
        assert_array_equal(data, X[:, j])
        assert_equal(n_nz, X.shape[0])


def test_csc_get_column():
    ds = CSCDataset(X_csc)
    for j in xrange(X.shape[1]):
        indices, data, n_nz = ds.get_column(j)
        for ii in xrange(n_nz):
            i = indices[ii]
            assert_equal(X[i, j], data[ii])


def test_contiguous_dot():
    ds = ContiguousDataset(X)
    assert_array_almost_equal(ds.dot(X2.T), np.dot(X, X2.T))


def test_fortran_dot():
    ds = FortranDataset(np.asfortranarray(X))
    assert_array_almost_equal(ds.dot(X2.T), np.dot(X, X2.T))


def test_csr_dot():
    ds = CSRDataset(X_csr)
    assert_array_almost_equal(ds.dot(X2.T), np.dot(X, X2.T))


def test_csc_dot():
    ds = CSCDataset(X_csc)
    assert_array_almost_equal(ds.dot(X2.T), np.dot(X, X2.T))


def check_kernel(K, kd):
    for i in xrange(K.shape[0]):
        indices, data, n_nz = kd.get_column(i)
        assert_array_almost_equal(K[i], data)
        assert_equal(n_nz, K.shape[0])


def test_dataset_linear_kernel():
    K = pairwise_kernels(X, metric="linear")
    kd = KernelDataset(X, X, "linear")
    check_kernel(K, kd)


def test_dataset_poly_kernel():
    K = pairwise_kernels(X, metric="poly", gamma=0.1, coef0=1, degree=4)
    kd = KernelDataset(X, X, "poly", gamma=0.1, coef0=1, degree=4)
    check_kernel(K, kd)


def test_dataset_rbf_kernel():
    K = pairwise_kernels(X, metric="rbf", gamma=0.1)
    kd = KernelDataset(X, X, "rbf", gamma=0.1)
    check_kernel(K, kd)


def test_kernel_dot():
    coef = rs.randn(X2.shape[0], 3)
    K = pairwise_kernels(X, X2, metric="rbf", gamma=0.1)
    kd = KernelDataset(X, X2, "rbf", gamma=0.1)
    assert_array_almost_equal(kd.dot(coef),
                              np.dot(K, coef))


def test_kernel_cache_add_remove():
    kds = KernelDataset(X, X, "rbf", gamma=0.1,
                        capacity=capacity, mb=0, verbose=0)

    for i in xrange(3):
        kds.add_sv(i)

    assert_equal(kds.n_sv(), 3)

    kds.remove_sv(1)
    assert_equal(kds.n_sv(), 2)


def test_kernel_cache_column():
    K = pairwise_kernels(X, metric="rbf", gamma=0.1)
    kds = KernelDataset(X, X, "rbf", gamma=0.1,
                        capacity=capacity, mb=0, verbose=0)

    # Compute a first column.
    data = kds.get_column(12)[1]
    assert_array_almost_equal(K[:, 12], data)
    assert_equal(kds.get_size(), 160)

    # Check that the works.
    data = kds.get_column(12)[1]
    assert_array_almost_equal(K[:, 12], data)
    assert_equal(kds.get_size(), 160)

    # Compute more columns.
    data = kds.get_column(13)[1]
    assert_array_almost_equal(K[:, 13], data)
    assert_equal(kds.get_size(), 320)

    data = kds.get_column(14)[1]
    assert_array_almost_equal(K[:, 14], data)
    assert_equal(kds.get_size(), 480)

    data = kds.get_column(15)[1]
    assert_array_almost_equal(K[:, 15], data)
    assert_equal(kds.get_size(), 640)

    # Maximum size reached.
    data = kds.get_column(16)[1]
    assert_array_almost_equal(K[:, 16], data)
    assert_equal(kds.get_size(), 480)

    # Check that cache works.
    data = kds.get_column(16)[1]
    assert_array_almost_equal(K[:, 16], data)
    assert_equal(kds.get_size(), 480)


def test_kernel_cache_column_sv():
    K = pairwise_kernels(X, metric="rbf", gamma=0.1)
    kds = KernelDataset(X, X, "rbf", gamma=0.1,
                        capacity=capacity, mb=0, verbose=0)

    size = 0

    # Add 3 SVs.
    kds.add_sv(6)
    kds.add_sv(12)
    kds.add_sv(3)
    # That's a limitation of the current implementation:
    # it allocates a full column.
    size += 20 * 8

    # Compute values.
    for i in xrange(2): # Do it twice to check that the cache works.
        data = kds.get_column_sv(7)
        assert_almost_equal(K[6, 7], data[6])
        assert_almost_equal(K[12, 7], data[12])
        assert_almost_equal(K[3, 7], data[3])
        assert_equal(size, kds.get_size())

    # Add one more SV.
    kds.add_sv(17)

    # Compute values.
    for i in xrange(2):
        data = kds.get_column_sv(7)
        assert_almost_equal(K[6, 7], data[6])
        assert_almost_equal(K[12, 7], data[12])
        assert_almost_equal(K[3, 7], data[3])
        assert_almost_equal(K[17, 7], data[17])
        assert_equal(size, kds.get_size())

    # Compute the entire same column.
    data = kds.get_column(7)[1]
    assert_array_almost_equal(K[:, 7], data)
    assert_equal(size, kds.get_size())

    # Compute an entire new column.
    data = kds.get_column(8)[1]
    size += 20 * 8
    assert_array_almost_equal(K[:, 8], data)
    assert_equal(size, kds.get_size())

    # Retrieve the SV only.
    data = kds.get_column_sv(8)
    assert_almost_equal(K[6, 8], data[6])
    assert_almost_equal(K[12, 8], data[12])
    assert_almost_equal(K[3, 8], data[3])
    assert_almost_equal(K[17, 8], data[17])
    assert_equal(size, kds.get_size())

    # Remove SV.
    kds.remove_sv(12)
    data = kds.get_column_sv(8)
    assert_almost_equal(K[6, 8], data[6])
    assert_almost_equal(K[12, 8], data[12])
    assert_almost_equal(K[3, 8], data[3])
    assert_almost_equal(K[17, 8], data[17])
    assert_equal(size, kds.get_size())

    for i in (12, 6, 3):
        kds.remove_sv(i)

    assert_equal(kds.n_sv(), 1)

    # Add back some new SV.
    kds.add_sv(15)
    assert_equal(kds.n_sv(), 2)

    data = kds.get_column_sv(8)
    assert_almost_equal(0, data[6])
    assert_almost_equal(0, data[12])
    assert_almost_equal(0, data[3])
    assert_almost_equal(K[15, 8], data[15])
    assert_almost_equal(K[17, 8], data[17])
    assert_equal(size, kds.get_size())

    # Remove non-existing column.
    kds.remove_column(19)
    assert_equal(size, kds.get_size())

    # Remove existing column.
    kds.remove_column(8)
    size -= 20 * 8
    assert_equal(size, kds.get_size())


def test_kernel_diag_linear():
    K = pairwise_kernels(X, metric="linear")
    kds = KernelDataset(X, X, "linear")
    out = kds.get_diag()
    good = K.flat[::K.shape[0] + 1]
    assert_array_almost_equal(out, good)


def test_kernel_diag_rbf():
    K = pairwise_kernels(X, metric="rbf", gamma=0.1)
    kds = KernelDataset(X, X, "rbf", gamma=0.1)
    out = kds.get_diag()
    good = K.flat[::K.shape[0] + 1]
    assert_array_almost_equal(out, good)


def test_kernel_get_element_linear():
    K = pairwise_kernels(X, metric="linear")
    kds = KernelDataset(X, X, "linear")
    for i in xrange(K.shape[0]):
        for j in xrange(K.shape[1]):
            assert_almost_equal(K[i, j], kds.get_element(i, j))


def test_kernel_get_element_rbf():
    K = pairwise_kernels(X, metric="rbf", gamma=0.1)
    kds = KernelDataset(X, X, "rbf", gamma=0.1)
    for i in xrange(K.shape[0]):
        for j in xrange(K.shape[1]):
            assert_almost_equal(K[i, j], kds.get_element(i, j))
