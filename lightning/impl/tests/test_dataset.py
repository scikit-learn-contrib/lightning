import pickle

import numpy as np
import pytest
import scipy.sparse as sp

from sklearn.datasets import make_classification
from sklearn.utils import check_random_state

from lightning.impl.dataset_fast import ContiguousDataset
from lightning.impl.dataset_fast import FortranDataset
from lightning.impl.dataset_fast import CSRDataset
from lightning.impl.dataset_fast import CSCDataset


@pytest.fixture(scope="module")
def test_data():
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

    return {
        "X": X,
        "X_csr": X_csr,
        "X_csc": X_csc,
        "contiguous_dataset": cds,
        "fortran_dataset": fds,
        "dataset_csr": csr_ds,
        "dataset_csc": csc_ds
    }


def test_contiguous_get_row(test_data):
    X = test_data["X"]
    cds = test_data["contiguous_dataset"]
    ind = np.arange(X.shape[1])
    for i in range(X.shape[0]):
        indices, data, n_nz = cds.get_row(i)
        np.testing.assert_array_equal(indices, ind)
        np.testing.assert_array_equal(data, X[i])
        assert n_nz == X.shape[1]


def test_csr_get_row(test_data):
    X = test_data["X"]
    csr_ds = test_data["dataset_csr"]
    for i in range(X.shape[0]):
        indices, data, n_nz = csr_ds.get_row(i)
        for jj in range(n_nz):
            j = indices[jj]
            assert X[i, j] == data[jj]


def test_fortran_get_column(test_data):
    X = test_data["X"]
    fds = test_data["fortran_dataset"]
    ind = np.arange(X.shape[0])
    for j in range(X.shape[1]):
        indices, data, n_nz = fds.get_column(j)
        np.testing.assert_array_equal(indices, ind)
        np.testing.assert_array_equal(data, X[:, j])
        assert n_nz == X.shape[0]


def test_csc_get_column(test_data):
    X = test_data["X"]
    csc_ds = test_data["dataset_csc"]
    for j in range(X.shape[1]):
        indices, data, n_nz = csc_ds.get_column(j)
        for ii in range(n_nz):
            i = indices[ii]
            assert X[i, j] == data[ii]


def test_picklable_datasets(test_data):
    # Test that the datasets are picklable.
    X = test_data["X"]
    for dataset in [
        test_data["contiguous_dataset"],
        test_data["dataset_csr"],
        test_data["fortran_dataset"],
        test_data["dataset_csc"]
    ]:
        pds = pickle.dumps(dataset)
        dataset = pickle.loads(pds)
        assert dataset.get_n_samples() == X.shape[0]
        assert dataset.get_n_features() == X.shape[1]
