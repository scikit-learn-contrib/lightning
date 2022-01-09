import numpy as np
import pytest

from scipy.linalg import svd, diagsvd

from lightning.classification import FistaClassifier
from lightning.regression import FistaRegressor
from lightning.impl.penalty import project_simplex, L1Penalty


@pytest.fixture(scope="module")
def bin_dense_train_data(bin_dense_train_data):
    bin_dense, bin_target = bin_dense_train_data
    bin_target = bin_target * 2 - 1
    return bin_dense, bin_target


@pytest.mark.parametrize("data", ["mult_dense_train_data", "mult_sparse_train_data"])
def test_fista_multiclass_l1l2(data, request):
    X, y = request.getfixturevalue(data)
    clf = FistaClassifier(max_iter=200, penalty="l1/l2", multiclass=True)
    clf.fit(X, y)
    np.testing.assert_almost_equal(clf.score(X, y), 0.99, 2)


@pytest.mark.parametrize("data", ["mult_dense_train_data", "mult_sparse_train_data"])
def test_fista_multiclass_l1l2_log(data, request):
    X, y = request.getfixturevalue(data)
    clf = FistaClassifier(max_iter=200, penalty="l1/l2", loss="log",
                          multiclass=True)
    clf.fit(X, y)
    np.testing.assert_almost_equal(clf.score(X, y), 0.90, 2)


@pytest.mark.parametrize("data", ["mult_dense_train_data", "mult_sparse_train_data"])
def test_fista_multiclass_l1l2_log_margin(data, request):
    X, y = request.getfixturevalue(data)
    clf = FistaClassifier(max_iter=200, penalty="l1/l2", loss="log_margin",
                          multiclass=True)
    clf.fit(X, y)
    np.testing.assert_almost_equal(clf.score(X, y), 0.93, 2)


@pytest.mark.parametrize("data", ["mult_dense_train_data", "mult_sparse_train_data"])
def test_fista_multiclass_l1(data, request):
    X, y = request.getfixturevalue(data)
    clf = FistaClassifier(max_iter=200, penalty="l1", multiclass=True)
    clf.fit(X, y)
    np.testing.assert_almost_equal(clf.score(X, y), 0.98, 2)


@pytest.mark.parametrize("data", ["mult_dense_train_data", "mult_sparse_train_data"])
def test_fista_multiclass_tv1d(data, request):
    X, y = request.getfixturevalue(data)
    clf = FistaClassifier(max_iter=200, penalty="tv1d", multiclass=True)
    clf.fit(X, y)
    np.testing.assert_almost_equal(clf.score(X, y), 0.97, 2)

    # adding a lot of regularization coef_ should be constant
    clf = FistaClassifier(max_iter=200, penalty="tv1d", multiclass=True, alpha=1e6)
    clf.fit(X, y)
    for i in range(clf.coef_.shape[0]):
        np.testing.assert_array_almost_equal(
            clf.coef_[i], np.mean(clf.coef_[i]) * np.ones(X.shape[1]))


@pytest.mark.parametrize("data", ["mult_dense_train_data", "mult_sparse_train_data"])
@pytest.mark.parametrize("penalty", ["l1/l2", "l1"])
def test_fista_multiclass_no_line_search(data, penalty, request):
    X, y = request.getfixturevalue(data)
    clf = FistaClassifier(max_iter=500, penalty=penalty, multiclass=True,
                          max_steps=0)
    clf.fit(X, y)
    np.testing.assert_almost_equal(clf.score(X, y), 0.94, 2)


@pytest.mark.parametrize("data", ["bin_dense_train_data", "bin_sparse_train_data"])
def test_fista_bin_l1(data, request):
    X, y = request.getfixturevalue(data)
    clf = FistaClassifier(max_iter=200, penalty="l1")
    clf.fit(X, y)
    np.testing.assert_almost_equal(clf.score(X, y), 1.0, 2)


@pytest.mark.parametrize("data", ["bin_dense_train_data", "bin_sparse_train_data"])
def test_fista_bin_l1_no_line_search(data, request):
    X, y = request.getfixturevalue(data)
    clf = FistaClassifier(max_iter=500, penalty="l1", max_steps=0)
    clf.fit(X, y)
    np.testing.assert_almost_equal(clf.score(X, y), 1.0, 2)


@pytest.mark.parametrize("data", ["mult_dense_train_data", "mult_sparse_train_data"])
def test_fista_multiclass_trace(data, request):
    X, y = request.getfixturevalue(data)
    clf = FistaClassifier(max_iter=100, penalty="trace", multiclass=True)
    clf.fit(X, y)
    np.testing.assert_almost_equal(clf.score(X, y), 0.96, 2)


def test_fista_bin_classes(bin_dense_train_data):
    X, y = bin_dense_train_data
    clf = FistaClassifier()
    clf.fit(X, y)
    assert list(clf.classes_) == [0, 1]


def test_fista_multiclass_classes(mult_dense_train_data):
    X, y = mult_dense_train_data
    clf = FistaClassifier()
    clf.fit(X, y)
    assert list(clf.classes_) == [0, 1, 2]


def test_fista_regression(bin_dense_train_data):
    X, y = bin_dense_train_data
    reg = FistaRegressor(max_iter=100, verbose=0)
    reg.fit(X, y)
    y_pred = np.sign(reg.predict(X))
    np.testing.assert_almost_equal(np.mean(y == y_pred), 0.985)


def test_fista_regression_simplex():
    rng = np.random.RandomState(0)
    w = project_simplex(rng.rand(10))
    X = rng.randn(1000, 10)
    y = np.dot(X, w)

    reg = FistaRegressor(penalty="simplex", max_iter=100, verbose=0)
    reg.fit(X, y)
    y_pred = reg.predict(X)
    error = np.sqrt(np.mean((y - y_pred) ** 2))
    np.testing.assert_almost_equal(error, 0.000, 3)
    assert np.all(reg.coef_ >= -1e-12)
    np.testing.assert_almost_equal(np.sum(reg.coef_), 1.0, 3)


def test_fista_regression_l1_ball():
    rng = np.random.RandomState(0)
    alpha = 5.0
    w = project_simplex(rng.randn(10), alpha)
    X = rng.randn(1000, 10)
    y = np.dot(X, w)

    reg = FistaRegressor(penalty="l1-ball", alpha=alpha, max_iter=100, verbose=0)
    reg.fit(X, y)
    y_pred = reg.predict(X)
    error = np.sqrt(np.mean((y - y_pred) ** 2))
    np.testing.assert_almost_equal(error, 0.000, 3)
    np.testing.assert_almost_equal(np.sum(np.abs(reg.coef_)), alpha, 3)


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

    X, Y, W = _make_data(200, 50, 30, 5)
    reg = FistaRegressor(max_iter=15, verbose=0)
    reg.fit(X, Y)
    Y_pred = reg.predict(X)
    error = (Y_pred - Y).ravel()
    error = np.dot(error, error)
    np.testing.assert_almost_equal(error, 77.44, 2)


@pytest.mark.parametrize("data", ["bin_dense_train_data", "bin_sparse_train_data"])
def test_fista_custom_prox(data, request):
    # test FISTA with a custom prox
    l1_pen = L1Penalty()
    X, y = request.getfixturevalue(data)
    clf = FistaClassifier(max_iter=500, penalty="l1", max_steps=0)
    clf.fit(X, y)

    clf2 = FistaClassifier(max_iter=500, penalty=l1_pen, max_steps=0)
    clf2.fit(X, y)
    np.testing.assert_array_almost_equal_nulp(clf.coef_.ravel(), clf2.coef_.ravel())
