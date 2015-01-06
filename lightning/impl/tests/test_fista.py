import numpy as np
import scipy.sparse as sp

from scipy.linalg import svd, diagsvd

from sklearn.utils.testing import assert_almost_equal

from sklearn.datasets import load_digits

from lightning.impl.datasets.samples_generator import make_classification
from lightning.classification import FistaClassifier
from lightning.regression import FistaRegressor

bin_dense, bin_target = make_classification(n_samples=200, n_features=100,
                                            n_informative=5,
                                            n_classes=2, random_state=0)
bin_target = bin_target * 2 - 1

mult_dense, mult_target = make_classification(n_samples=300, n_features=100,
                                              n_informative=5,
                                              n_classes=3, random_state=0)
bin_csr = sp.csr_matrix(bin_dense)
mult_csr = sp.csr_matrix(mult_dense)
digit = load_digits(2)


def test_fista_multiclass_l1l2():
    for data in (mult_dense, mult_csr):
        clf = FistaClassifier(max_iter=200, penalty="l1/l2", multiclass=True)
        clf.fit(data, mult_target)
        assert_almost_equal(clf.score(data, mult_target), 0.99, 2)


def test_fista_multiclass_l1l2_log():
    for data in (mult_dense, mult_csr):
        clf = FistaClassifier(max_iter=200, penalty="l1/l2", loss="log",
                              multiclass=True)
        clf.fit(data, mult_target)
        assert_almost_equal(clf.score(data, mult_target), 0.90, 2)

def test_fista_multiclass_l1l2_log_margin():
    for data in (mult_dense, mult_csr):
        clf = FistaClassifier(max_iter=200, penalty="l1/l2", loss="log_margin",
                              multiclass=True)
        clf.fit(data, mult_target)
        assert_almost_equal(clf.score(data, mult_target), 0.95, 2)


def test_fista_multiclass_l1():
    for data in (mult_dense, mult_csr):
        clf = FistaClassifier(max_iter=200, penalty="l1", multiclass=True)
        clf.fit(data, mult_target)
        assert_almost_equal(clf.score(data, mult_target), 0.98)


def test_fista_multiclass_l1l2_no_line_search():
    for data in (mult_dense, mult_csr):
        clf = FistaClassifier(max_iter=500, penalty="l1/l2", multiclass=True,
                              max_steps=0)
        clf.fit(data, mult_target)
        assert_almost_equal(clf.score(data, mult_target), 0.96, 2)


def test_fista_multiclass_l1_no_line_search():
    for data in (mult_dense, mult_csr):
        clf = FistaClassifier(max_iter=500, penalty="l1", multiclass=True,
                              max_steps=0)
        clf.fit(data, mult_target)
        assert_almost_equal(clf.score(data, mult_target), 0.95, 2)


def test_fista_bin_l1():
    for data in (bin_dense, bin_csr):
        clf = FistaClassifier(max_iter=200, penalty="l1")
        clf.fit(data, bin_target)
        assert_almost_equal(clf.score(data, bin_target), 1.0, 2)


def test_fista_bin_l1_no_line_search():
    for data in (bin_dense, bin_csr):
        clf = FistaClassifier(max_iter=500, penalty="l1", max_steps=0)
        clf.fit(data, bin_target)
        assert_almost_equal(clf.score(data, bin_target), 1.0, 2)


def test_fista_multiclass_trace():
    for data in (mult_dense, mult_csr):
        clf = FistaClassifier(max_iter=100, penalty="trace", multiclass=True)
        clf.fit(data, mult_target)
        assert_almost_equal(clf.score(data, mult_target), 0.98, 2)


def test_fista_regression():
    reg = FistaRegressor(max_iter=100, verbose=0)
    reg.fit(bin_dense, bin_target)
    y_pred = np.sign(reg.predict(bin_dense))
    assert_almost_equal(np.mean(bin_target == y_pred), 0.985)


def test_fista_regression_trace():
    rng = np.random.RandomState(0)
    def _make_data(n_samples, n_features, n_tasks, n_components):
        W = rng.rand(n_tasks, n_features) - 0.5
        U, S, V = svd(W, full_matrices=True)
        S[n_components:] = 0
        S = diagsvd(S, U.shape[0], V.shape[0])
        W = np.dot(np.dot(U, S), V)
        X = rng.rand(n_samples, n_features) - 0.5
        Y = np.dot(X, W.T)
        return X, Y, W

    X, Y, W = _make_data(200, 50,30, 5)
    reg = FistaRegressor(max_iter=15, verbose=0)
    reg.fit(X, Y)
    Y_pred = reg.predict(X)
    error = (Y_pred - Y).ravel()
    error = np.dot(error, error)
    assert_almost_equal(error, 77.45, 2)
