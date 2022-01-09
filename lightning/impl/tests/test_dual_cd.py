import numpy as np
import pytest

from sklearn.metrics.pairwise import linear_kernel
from sklearn.datasets import make_regression

from lightning.impl.dual_cd import LinearSVC
from lightning.impl.dual_cd import LinearSVR
from lightning.impl.dual_cd_fast import sparse_dot
from lightning.impl.dataset_fast import get_dataset


@pytest.fixture(scope="module")
def reg_train_data():
    reg_dense, reg_target = make_regression(n_samples=200, n_features=100,
                                            n_informative=5, random_state=0)
    return reg_dense, reg_target


@pytest.mark.parametrize("data", ["bin_dense_train_data", "bin_sparse_train_data"])
def test_sparse_dot(data, request):
    X, _ = request.getfixturevalue(data)
    K = linear_kernel(X)
    K2 = np.zeros_like(K)
    ds = get_dataset(X)

    for i in range(X.shape[0]):
        for j in range(i, X.shape[0]):
            K2[i, j] = sparse_dot(ds, i, j)
            K2[j, i] = K[i, j]

    np.testing.assert_array_almost_equal(K, K2)


@pytest.mark.parametrize("data", ["bin_dense_train_data", "bin_sparse_train_data"])
@pytest.mark.parametrize("loss", ["l1", "l2"])
def test_fit_linear_binary(data, loss, request):
    X, y = request.getfixturevalue(data)
    clf = LinearSVC(loss=loss, random_state=0, max_iter=10)
    clf.fit(X, y)
    assert list(clf.classes_) == [0, 1]
    assert clf.score(X, y) == 1.0
    y_pred = clf.decision_function(X).ravel()


@pytest.mark.parametrize("data", ["bin_dense_train_data", "bin_sparse_train_data"])
@pytest.mark.parametrize("loss", ["l1", "l2"])
def test_fit_linear_binary_auc(data, loss, request):
    X, y = request.getfixturevalue(data)
    clf = LinearSVC(loss=loss, criterion="auc", random_state=0, max_iter=25)
    clf.fit(X, y)
    assert clf.score(X, y) == 1.0


@pytest.mark.parametrize("data", ["mult_dense_train_data", "mult_sparse_train_data"])
def test_fit_linear_multi(data, request):
    X, y = request.getfixturevalue(data)
    clf = LinearSVC(random_state=0)
    clf.fit(X, y)
    assert list(clf.classes_) == [0, 1, 2]
    y_pred = clf.predict(X)
    acc = np.mean(y_pred == y)
    assert acc > 0.85


@pytest.mark.parametrize("C", [0.1, 0.2])
def test_warm_start(bin_dense_train_data, C):
    bin_dense, bin_target = bin_dense_train_data
    clf = LinearSVC(warm_start=True, loss="l1", random_state=0, max_iter=100)
    clf.C = C
    clf.fit(bin_dense, bin_target)
    acc = clf.score(bin_dense, bin_target)
    assert acc > 0.99


@pytest.mark.parametrize("fit_intercept", [True, False])
@pytest.mark.parametrize("loss", ["epsilon_insensitive", "l2"])
def test_linear_svr(reg_train_data, fit_intercept, loss):
    reg_dense, reg_target = reg_train_data
    reg = LinearSVR(random_state=0, fit_intercept=fit_intercept, loss=loss)
    reg.fit(reg_dense, reg_target)
    assert reg.score(reg_dense, reg_target) > 0.99


@pytest.mark.parametrize("C, warm_start", [(1e-3, True), (1, False)])
def test_linear_svr_warm_start(reg_train_data, C, warm_start):
    reg_dense, reg_target = reg_train_data
    reg = LinearSVR(C=C, random_state=0, warm_start=warm_start)
    reg.fit(reg_dense, reg_target)
    target_score = 0.99 if C == 1 else 0.96
    assert reg.score(reg_dense, reg_target) > target_score
