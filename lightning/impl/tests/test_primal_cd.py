import numpy as np
import pytest

from sklearn.datasets import load_digits
from sklearn.preprocessing import LabelBinarizer

from lightning.impl.primal_cd import CDClassifier, CDRegressor
from lightning.impl.tests.utils import check_predict_proba


@pytest.fixture(scope="module")
def train_data():
    digit = load_digits(n_class=2)
    return digit.data, digit.target


def test_fit_linear_binary_l1r(bin_dense_train_data):
    bin_dense, bin_target = bin_dense_train_data
    clf = CDClassifier(C=1.0, random_state=0, penalty="l1")
    clf.fit(bin_dense, bin_target)
    assert not hasattr(clf, 'predict_proba')
    acc = clf.score(bin_dense, bin_target)
    np.testing.assert_almost_equal(acc, 1.0)
    n_nz = clf.n_nonzero()
    perc = clf.n_nonzero(percentage=True)
    assert perc == n_nz / bin_dense.shape[1]

    clf = CDClassifier(C=0.1, random_state=0, penalty="l1")
    clf.fit(bin_dense, bin_target)
    acc = clf.score(bin_dense, bin_target)
    np.testing.assert_almost_equal(acc, 0.97)
    n_nz2 = clf.n_nonzero()
    perc2 = clf.n_nonzero(percentage=True)
    assert perc2 == n_nz2 / bin_dense.shape[1]

    assert n_nz > n_nz2


def test_fit_linear_binary_l1r_smooth_hinge(bin_dense_train_data):
    bin_dense, bin_target = bin_dense_train_data
    clf = CDClassifier(C=1.0, loss="smooth_hinge", random_state=0, penalty="l1")
    clf.fit(bin_dense, bin_target)
    assert not hasattr(clf, 'predict_proba')
    acc = clf.score(bin_dense, bin_target)
    np.testing.assert_almost_equal(acc, 1.0)


def test_fit_linear_binary_l1r_no_linesearch(bin_dense_train_data):
    bin_dense, bin_target = bin_dense_train_data
    clf = CDClassifier(C=1.0, selection="uniform", max_steps=0,
                       random_state=0, penalty="l1")
    clf.fit(bin_dense, bin_target)
    acc = clf.score(bin_dense, bin_target)
    np.testing.assert_almost_equal(acc, 1.0)


@pytest.mark.parametrize("shrinking", [True, False])
def test_l1r_shrinking(bin_dense_train_data, shrinking):
    bin_dense, bin_target = bin_dense_train_data
    clf = CDClassifier(C=0.5, penalty="l1", random_state=0,
                       shrinking=shrinking)
    clf.fit(bin_dense, bin_target)
    assert clf.score(bin_dense, bin_target) == 1.0


def test_warm_start_l1r(bin_dense_train_data):
    bin_dense, bin_target = bin_dense_train_data
    clf = CDClassifier(warm_start=True, random_state=0, penalty="l1")

    clf.C = 0.1
    clf.fit(bin_dense, bin_target)
    n_nz = clf.n_nonzero()

    clf.C = 0.2
    clf.fit(bin_dense, bin_target)
    n_nz2 = clf.n_nonzero()

    assert n_nz < n_nz2


def test_warm_start_l1r_regression(bin_dense_train_data):
    bin_dense, bin_target = bin_dense_train_data
    clf = CDRegressor(warm_start=True, random_state=0, penalty="l1")

    clf.C = 0.1
    clf.fit(bin_dense, bin_target)
    n_nz = clf.n_nonzero()

    clf.C = 0.2
    clf.fit(bin_dense, bin_target)
    n_nz2 = clf.n_nonzero()

    assert n_nz < n_nz2


def test_fit_linear_binary_l1r_log_loss(bin_dense_train_data):
    bin_dense, bin_target = bin_dense_train_data
    clf = CDClassifier(C=1.0, random_state=0, penalty="l1", loss="log")
    clf.fit(bin_dense, bin_target)
    check_predict_proba(clf, bin_dense)
    acc = clf.score(bin_dense, bin_target)
    np.testing.assert_almost_equal(acc, 0.995)


def test_fit_linear_binary_l1r_log_loss_no_linesearch(bin_dense_train_data):
    bin_dense, bin_target = bin_dense_train_data
    clf = CDClassifier(C=1.0, max_steps=0, random_state=0,
                       selection="uniform", penalty="l1", loss="log")
    clf.fit(bin_dense, bin_target)
    acc = clf.score(bin_dense, bin_target)
    np.testing.assert_almost_equal(acc, 0.995)


def test_fit_linear_binary_l2r(bin_dense_train_data):
    bin_dense, bin_target = bin_dense_train_data
    clf = CDClassifier(C=1.0, random_state=0, penalty="l2")
    clf.fit(bin_dense, bin_target)
    acc = clf.score(bin_dense, bin_target)
    np.testing.assert_almost_equal(acc, 1.0)


def test_fit_linear_binary_l2r_log(bin_dense_train_data):
    bin_dense, bin_target = bin_dense_train_data
    clf = CDClassifier(C=1.0, random_state=0, penalty="l2", loss="log",
                       max_iter=5)
    clf.fit(bin_dense, bin_target)
    acc = clf.score(bin_dense, bin_target)
    np.testing.assert_almost_equal(acc, 1.0)


def test_fit_linear_binary_l2r_modified_huber(bin_dense_train_data):
    bin_dense, bin_target = bin_dense_train_data
    clf = CDClassifier(C=1.0, random_state=0, penalty="l2",
                       loss="modified_huber")
    clf.fit(bin_dense, bin_target)
    check_predict_proba(clf, bin_dense)
    acc = clf.score(bin_dense, bin_target)
    np.testing.assert_almost_equal(acc, 1.0)


def test_fit_linear_multi_l2r(mult_dense_train_data):
    mult_dense, mult_target = mult_dense_train_data
    clf = CDClassifier(C=1.0, random_state=0, penalty="l2")
    clf.fit(mult_dense, mult_target)
    acc = clf.score(mult_dense, mult_target)
    np.testing.assert_almost_equal(acc, 0.8833, 4)


def test_warm_start_l2r(bin_dense_train_data):
    bin_dense, bin_target = bin_dense_train_data
    clf = CDClassifier(warm_start=True, random_state=0, penalty="l2")

    clf.C = 0.1
    clf.fit(bin_dense, bin_target)
    np.testing.assert_almost_equal(clf.score(bin_dense, bin_target), 1.0)

    clf.C = 0.2
    clf.fit(bin_dense, bin_target)
    np.testing.assert_almost_equal(clf.score(bin_dense, bin_target), 1.0)


@pytest.mark.parametrize("warm_debiasing", [True, False])
def test_debiasing_l1(bin_dense_train_data, warm_debiasing):
    bin_dense, bin_target = bin_dense_train_data
    clf = CDClassifier(penalty="l1", debiasing=True,
                       warm_debiasing=warm_debiasing,
                       C=0.05, Cd=1.0, max_iter=10, random_state=0)
    clf.fit(bin_dense, bin_target)
    assert clf.n_nonzero() == 22
    np.testing.assert_almost_equal(clf.score(bin_dense, bin_target), 0.955, 3)


@pytest.mark.parametrize("warm_debiasing", [True, False])
def test_debiasing_l1l2(mult_sparse_train_data, warm_debiasing):
    mult_sparse, mult_target = mult_sparse_train_data
    clf = CDClassifier(penalty="l1/l2", loss="squared_hinge",
                       multiclass=False,
                       debiasing=True,
                       warm_debiasing=warm_debiasing,
                       max_iter=20, C=0.01, random_state=0)
    clf.fit(mult_sparse, mult_target)
    assert clf.score(mult_sparse, mult_target) > 0.75
    assert clf.n_nonzero(percentage=True) == 0.08


def test_debiasing_warm_start(bin_dense_train_data):
    bin_dense, bin_target = bin_dense_train_data
    clf = CDClassifier(penalty="l1", max_iter=10,
                       warm_start=True, random_state=0)
    clf.C = 0.5
    clf.fit(bin_dense, bin_target)
    assert clf.n_nonzero() == 74
    np.testing.assert_almost_equal(clf.score(bin_dense, bin_target), 1.0)

    clf.C = 1.0
    clf.fit(bin_dense, bin_target)
    # FIXME: not the same sparsity as without warm start...
    assert clf.n_nonzero() == 77
    np.testing.assert_almost_equal(clf.score(bin_dense, bin_target), 1.0)


def test_empty_model(bin_dense_train_data):
    bin_dense, bin_target = bin_dense_train_data
    clf = CDClassifier(C=1e-5, penalty="l1")
    clf.fit(bin_dense, bin_target)
    assert clf.n_nonzero() == 0
    acc = clf.score(bin_dense, bin_target)
    assert acc == 0.5

    clf = CDClassifier(C=1e-5, penalty="l1", debiasing=True)
    clf.fit(bin_dense, bin_target)
    assert clf.n_nonzero() == 0
    acc = clf.score(bin_dense, bin_target)
    assert acc == 0.5


def test_fit_squared_loss(bin_dense_train_data):
    bin_dense, bin_target = bin_dense_train_data
    clf = CDClassifier(C=1.0, random_state=0, penalty="l2",
                       loss="squared", max_iter=100)
    clf.fit(bin_dense, bin_target)
    np.testing.assert_almost_equal(clf.score(bin_dense, bin_target), 0.99)
    y = bin_target.copy()
    y[y == 0] = -1
    np.testing.assert_array_almost_equal(np.dot(bin_dense, clf.coef_.ravel()) - y,
                                         clf.errors_.ravel())


def test_fit_squared_loss_l1(bin_dense_train_data):
    bin_dense, bin_target = bin_dense_train_data
    clf = CDClassifier(C=0.5, random_state=0, penalty="l1",
                       loss="squared", max_iter=100, shrinking=False)
    clf.fit(bin_dense, bin_target)
    np.testing.assert_almost_equal(clf.score(bin_dense, bin_target), 0.985, 3)
    y = bin_target.copy()
    y[y == 0] = -1
    np.testing.assert_array_almost_equal(np.dot(bin_dense, clf.coef_.ravel()) - y,
                                         clf.errors_.ravel())
    n_nz = clf.n_nonzero()
    assert n_nz == 89


@pytest.mark.parametrize("data", [mult_dense_train_data, mult_sparse_train_data])
def test_l1l2_multiclass_log_loss(data):
    X, y = data
    clf = CDClassifier(penalty="l1/l2", loss="log", multiclass=True,
                       max_steps=30, max_iter=5, C=1.0, random_state=0)
    clf.fit(X, y)
    np.testing.assert_almost_equal(clf.score(X, y), 0.8766, 3)
    df = clf.decision_function(X)
    sel = np.array([df[i, int(y[i])] for i in range(df.shape[0])])
    df -= sel[:, np.newaxis]
    df = np.exp(df)
    np.testing.assert_array_almost_equal(clf.errors_, df.T)
    for i in range(X.shape[0]):
        np.testing.assert_almost_equal(clf.errors_[y[i], i], 1.0)
    nz = np.sum(clf.coef_ != 0)
    assert nz == 297

    clf = CDClassifier(penalty="l1/l2", loss="log", multiclass=True,
                       max_steps=30, max_iter=5, C=0.3, random_state=0)
    clf.fit(X, y)
    np.testing.assert_almost_equal(clf.score(X, y), 0.8566, 3)
    nz = np.sum(clf.coef_ != 0)
    assert nz == 213
    assert nz % 3 == 0  # should be a multiple of n_classes


def test_l1l2_multiclass_log_loss_no_linesearch(mult_sparse_train_data):
    mult_sparse, mult_target = mult_sparse_train_data
    clf = CDClassifier(penalty="l1/l2", loss="log", multiclass=True,
                       selection="uniform", max_steps=0,
                       max_iter=30, C=1.0, random_state=0)
    clf.fit(mult_sparse, mult_target)
    np.testing.assert_almost_equal(clf.score(mult_sparse, mult_target), 0.88, 3)
    nz = np.sum(clf.coef_ != 0)
    assert nz == 297


@pytest.mark.parametrize("data", [mult_dense_train_data, mult_sparse_train_data])
def test_l1l2_multiclass_squared_hinge_loss(data):
    X, y = data
    clf = CDClassifier(penalty="l1/l2", loss="squared_hinge",
                       multiclass=True,
                       max_iter=20, C=1.0, random_state=0)
    clf.fit(X, y)
    np.testing.assert_almost_equal(clf.score(X, y), 0.913, 3)
    df = clf.decision_function(X)
    n_samples, n_vectors = df.shape
    diff = np.zeros_like(clf.errors_)
    for i in range(n_samples):
        for k in range(n_vectors):
            diff[k, i] = 1 - (df[i, y[i]] - df[i, k])
    np.testing.assert_array_almost_equal(clf.errors_, diff)
    assert np.sum(clf.coef_ != 0) == 300

    clf = CDClassifier(penalty="l1/l2", loss="squared_hinge",
                       multiclass=True,
                       max_iter=20, C=0.05, random_state=0)
    clf.fit(X, y)
    np.testing.assert_almost_equal(clf.score(X, y), 0.83, 3)
    nz = np.sum(clf.coef_ != 0)
    assert nz == 207
    assert nz % 3 == 0  # should be a multiple of n_classes


def test_l1l2_multiclass_squared_hinge_loss_no_linesearch(mult_sparse_train_data):
    mult_sparse, mult_target = mult_sparse_train_data
    clf = CDClassifier(penalty="l1/l2", loss="squared_hinge",
                       multiclass=True, shrinking=False, selection="uniform",
                       max_steps=0, max_iter=200, C=1.0, random_state=0)
    clf.fit(mult_sparse, mult_target)
    np.testing.assert_almost_equal(clf.score(mult_sparse, mult_target), 0.9166, 3)
    df = clf.decision_function(mult_sparse)
    n_samples, n_vectors = df.shape
    diff = np.zeros_like(clf.errors_)
    for i in range(n_samples):
        for k in range(n_vectors):
            diff[k, i] = 1 - (df[i, mult_target[i]] - df[i, k])
    np.testing.assert_array_almost_equal(clf.errors_, diff)
    assert np.sum(clf.coef_ != 0) == 300

    clf = CDClassifier(penalty="l1/l2", loss="squared_hinge",
                       multiclass=True,
                       max_iter=20, C=0.05, random_state=0)
    clf.fit(mult_sparse, mult_target)
    np.testing.assert_almost_equal(clf.score(mult_sparse, mult_target), 0.83, 3)
    nz = np.sum(clf.coef_ != 0)
    assert nz == 207
    assert nz % 3 == 0  # should be a multiple of n_classes


def test_l1l2_multi_task_squared_hinge_loss(mult_dense_train_data):
    mult_dense, mult_target = mult_dense_train_data
    Y = LabelBinarizer(neg_label=-1).fit_transform(mult_target)
    clf = CDClassifier(penalty="l1/l2", loss="squared_hinge",
                       multiclass=False,
                       max_iter=20, C=5.0, random_state=0)
    clf.fit(mult_dense, mult_target)
    df = clf.decision_function(mult_dense)
    np.testing.assert_array_almost_equal(clf.errors_.T, 1 - Y * df)
    np.testing.assert_almost_equal(clf.score(mult_dense, mult_target), 0.8633, 3)
    nz = np.sum(clf.coef_ != 0)
    assert nz == 300

    clf = CDClassifier(penalty="l1/l2", loss="squared_hinge",
                       multiclass=False,
                       max_iter=20, C=0.05, random_state=0)
    clf.fit(mult_dense, mult_target)
    np.testing.assert_almost_equal(clf.score(mult_dense, mult_target), 0.8266, 3)
    nz = np.sum(clf.coef_ != 0)
    assert nz == 231


def test_l1l2_multi_task_log_loss(mult_dense_train_data):
    mult_dense, mult_target = mult_dense_train_data
    clf = CDClassifier(penalty="l1/l2", loss="log",
                       multiclass=False,
                       max_steps=30,
                       max_iter=20, C=5.0, random_state=0)
    clf.fit(mult_dense, mult_target)
    np.testing.assert_almost_equal(clf.score(mult_dense, mult_target), 0.8633, 3)


def test_l1l2_multi_task_square_loss(mult_dense_train_data):
    mult_dense, mult_target = mult_dense_train_data
    clf = CDClassifier(penalty="l1/l2", loss="squared",
                       multiclass=False,
                       max_iter=20, C=5.0, random_state=0)
    clf.fit(mult_dense, mult_target)
    np.testing.assert_almost_equal(clf.score(mult_dense, mult_target), 0.8066, 3)


def test_fit_reg_squared_l2(train_data):
    X, y = train_data
    clf = CDRegressor(C=1.0, random_state=0, penalty="l2",
                      loss="squared", max_iter=100)
    clf.fit(X, y)
    y_pred = (clf.predict(X) > 0.5).astype(int)
    acc = np.mean(y == y_pred)
    np.testing.assert_almost_equal(acc, 1.0, 3)


def test_fit_reg_squared_l1(train_data):
    X, y = train_data
    clf = CDRegressor(C=1.0, random_state=0, penalty="l1",
                      loss="squared", max_iter=100)
    clf.fit(X, y)
    y_pred = (clf.predict(X) > 0.5).astype(int)
    acc = np.mean(y == y_pred)
    np.testing.assert_almost_equal(acc, 1.0, 3)


def test_fit_reg_squared_multiple_outputs(train_data):
    X, y = train_data
    reg = CDRegressor(C=1.0, random_state=0, penalty="l2",
                      loss="squared", max_iter=100)
    Y = np.zeros((len(y), 2))
    Y[:, 0] = y
    Y[:, 1] = y
    reg.fit(X, Y)
    y_pred = reg.predict(X)
    assert y_pred.shape[0] == len(y)
    assert y_pred.shape[1] == 2


def test_fit_reg_squared_multiple_outputs(mult_dense_train_data):
    mult_dense, mult_target = mult_dense_train_data
    reg = CDRegressor(C=0.05, random_state=0, penalty="l1/l2",
                      loss="squared", max_iter=100)
    lb = LabelBinarizer()
    Y = lb.fit_transform(mult_target)
    reg.fit(mult_dense, Y)
    y_pred = lb.inverse_transform(reg.predict(mult_dense))
    np.testing.assert_almost_equal(np.mean(y_pred == mult_target), 0.797, 3)
    np.testing.assert_almost_equal(reg.n_nonzero(percentage=True), 0.5)


@pytest.mark.parametrize("penalty", ["l1", "l2"])
def test_multiclass_error_nongrouplasso(mult_dense_train_data, penalty):
    mult_dense, mult_target = mult_dense_train_data
    clf = CDClassifier(multiclass=True, penalty=penalty)
    with pytest.raises(NotImplementedError):
        clf.fit(mult_dense, mult_target)


def test_bin_classes(bin_dense_train_data):
    bin_dense, bin_target = bin_dense_train_data
    clf = CDClassifier()
    clf.fit(bin_dense, bin_target)
    assert list(clf.classes_) == [0, 1]


def test_multiclass_classes(mult_dense_train_data):
    mult_dense, mult_target = mult_dense_train_data
    clf = CDClassifier()
    clf.fit(mult_dense, mult_target)
    assert list(clf.classes_) == [0, 1, 2]
