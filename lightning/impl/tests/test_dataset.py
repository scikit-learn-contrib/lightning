import pickle
import numpy as np
import scipy.sparse as sp

from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_equal

from sklearn.datasets.samples_generator import make_classification
from sklearn.utils import check_random_state

from lightning.impl.dataset_fast import ContiguousDataset
from lightning.impl.dataset_fast import FortranDataset
from lightning.impl.dataset_fast import CSRDataset
from lightning.impl.dataset_fast import CSCDataset

# Create test datasets.
X, _ = make_classification(n_samples=20, n_features=100,
                           n_informative=5, n_classes=2, random_state=0)
X2, _ = make_classification(n_samples=10, n_features=100,
                            n_informative=5, n_classes=2, random_state=0)

# Sparsify datasets.
X[X < 0.3] = 0

X_csr = sp.csr_matrix(X)
X_csc = sp.csc_matrix(X)

rs = check_random_state(0)
cds = ContiguousDataset(X)
fds = FortranDataset(np.asfortranarray(X))
csr_ds = CSRDataset(X_csr)
csc_ds = CSCDataset(X_csc)

# fix for missing xrange in Python3
try:
    xrange
except NameError:
    xrange = range


def test_contiguous_get_row():
    ind = np.arange(X.shape[1])
    for i in xrange(X.shape[0]):
        indices, data, n_nz = cds.get_row(i)
        assert_array_equal(indices, ind)
        assert_array_equal(data, X[i])
        assert_equal(n_nz, X.shape[1])


def test_csr_get_row():
    for i in xrange(X.shape[0]):
        indices, data, n_nz = csr_ds.get_row(i)
        for jj in xrange(n_nz):
            j = indices[jj]
            assert_equal(X[i, j], data[jj])


def test_fortran_get_column():
    ind = np.arange(X.shape[0])
    for j in xrange(X.shape[1]):
        indices, data, n_nz = fds.get_column(j)
        assert_array_equal(indices, ind)
        assert_array_equal(data, X[:, j])
        assert_equal(n_nz, X.shape[0])


def test_csc_get_column():
    for j in xrange(X.shape[1]):
        indices, data, n_nz = csc_ds.get_column(j)
        for ii in xrange(n_nz):
            i = indices[ii]
            assert_equal(X[i, j], data[ii])


def test_picklable_datasets():
    """Test that the datasets are picklable."""

    for dataset in [cds, csr_ds, fds, csc_ds]:
        pds = pickle.dumps(dataset)
        dataset = pickle.loads(pds)
        assert_equal(dataset.get_n_samples(), X.shape[0])
        assert_equal(dataset.get_n_features(), X.shape[1])
