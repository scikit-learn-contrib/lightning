import pytest
import scipy.sparse as sp

from sklearn.datasets import load_iris

from lightning.impl.datasets.samples_generator import make_classification


@pytest.fixture(scope="module")
def train_data():
    iris = load_iris()
    X, y = iris.data, iris.target
    return X, y


@pytest.fixture(scope="module")
def bin_train_data(train_data):
    X, y = train_data
    X_bin = X[y <= 1]
    y_bin = y[y <= 1] * 2 - 1
    return X_bin, y_bin


@pytest.fixture(scope="module")
def bin_dense_train_data():
    bin_dense, bin_target = make_classification(n_samples=200, n_features=100,
                                                n_informative=5,
                                                n_classes=2, random_state=0)
    return bin_dense, bin_target


@pytest.fixture(scope="module")
def bin_sparse_train_data(bin_dense_train_data):
    bin_dense, bin_target = bin_dense, bin_target
    bin_csr = sp.csr_matrix(bin_dense)
    return bin_csr, bin_target


@pytest.fixture(scope="module")
def mult_dense_train_data():
    mult_dense, mult_target = make_classification(n_samples=300, n_features=100,
                                                  n_informative=5,
                                                  n_classes=3, random_state=0)
    return mult_dense, mult_target


@pytest.fixture(scope="module")
def mult_sparse_train_data(mult_dense_train_data):
    mult_dense, mult_target = mult_dense_train_data
    mult_sparse = sp.csr_matrix(mult_dense)
    return mult_sparse, mult_target
