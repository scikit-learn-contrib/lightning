import numpy as np
import pytest

from sklearn.datasets import make_regression

from lightning.impl.datasets.samples_generator import make_nn_regression
from lightning.impl.sgd import SGDClassifier
from lightning.impl.sgd import SGDRegressor
from lightning.impl.tests.utils import check_predict_proba


@pytest.fixture(scope="module")
def reg_train_data():
    X, y = make_regression(n_samples=100, n_features=10, n_informative=8,
                           random_state=0)
    return X, y


@pytest.fixture(scope="module")
def reg_nn_train_data():
    X, y, _ = make_nn_regression(n_samples=100, n_features=10, n_informative=8,
                                 random_state=0)
    return X, y


@pytest.mark.parametrize("data", [bin_dense_train_data, bin_sparse_train_data])
@pytest.mark.parametrize("clf", [SGDClassifier(random_state=0, loss="hinge",
                                               fit_intercept=True, learning_rate="pegasos"),
                                 SGDClassifier(random_state=0, loss="hinge",
                                               fit_intercept=False, learning_rate="pegasos"),
                                 SGDClassifier(random_state=0, loss="hinge",
                                               fit_intercept=True, learning_rate="invscaling"),
                                 SGDClassifier(random_state=0, loss="hinge",
                                               fit_intercept=True, learning_rate="constant"),
                                 SGDClassifier(random_state=0, loss="squared_hinge",
                                               eta0=1e-2,
                                               fit_intercept=True, learning_rate="constant"),
                                 SGDClassifier(random_state=0, loss="log",
                                               fit_intercept=True, learning_rate="constant"),
                                 SGDClassifier(random_state=0, loss="modified_huber",
                                               fit_intercept=True, learning_rate="constant")])
def test_binary_linear_sgd(data, clf):
    X, y = data
    clf.fit(X, y)
    assert clf.score(X, y) > 0.934
    assert list(clf.classes_) == [0, 1]
    if clf.loss in {'log', 'modified_huber'}:
        check_predict_proba(clf, X)
    else:
        assert not hasattr(clf, 'predict_proba')


def test_multiclass_sgd(mult_dense_train_data):
    mult_dense, mult_target = mult_dense_train_data
    clf = SGDClassifier(random_state=0)
    clf.fit(mult_dense, mult_target)
    assert clf.score(mult_dense, mult_target) > 0.80
    assert list(clf.classes_) == [0, 1, 2]


@pytest.mark.parametrize("data", [mult_dense_train_data, mult_sparse_train_data])
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_multiclass_hinge_sgd(data, fit_intercept):
    X, y = data
    clf = SGDClassifier(loss="hinge", multiclass=True,
                        fit_intercept=fit_intercept, random_state=0)
    clf.fit(X, y)
    assert clf.score(X, y) > 0.78


@pytest.mark.parametrize("data", [mult_dense_train_data, mult_sparse_train_data])
def test_multiclass_hinge_sgd_l1l2(data):
    X, y = data
    clf = SGDClassifier(loss="hinge", penalty="l1/l2",
                        multiclass=True, random_state=0)
    clf.fit(X, y)
    assert clf.score(X, y) > 0.75


@pytest.mark.parametrize("data", [mult_dense_train_data, mult_sparse_train_data])
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_multiclass_squared_hinge_sgd(data, fit_intercept):
    X, y = data
    clf = SGDClassifier(loss="squared_hinge", multiclass=True,
                        learning_rate="constant", eta0=1e-3,
                        fit_intercept=fit_intercept, random_state=0)
    clf.fit(X, y)
    assert clf.score(X, y) > 0.78


@pytest.mark.parametrize("data", [mult_dense_train_data, mult_sparse_train_data])
@pytest.mark.parametrize("fit_intercept", [True, False])
def test_multiclass_log_sgd(data, fit_intercept):
    X, y = data
    clf = SGDClassifier(loss="log", multiclass=True,
                        fit_intercept=fit_intercept,
                        random_state=0)
    clf.fit(X, y)
    assert clf.score(X, y) > 0.78


def test_regression_squared_loss(reg_train_data):
    X, y = reg_train_data
    reg = SGDRegressor(loss="squared", penalty="l2", learning_rate="constant",
                       eta0=1e-2, random_state=0)

    reg.fit(X, y)
    pred = reg.predict(X)
    np.testing.assert_almost_equal(np.mean((pred - y) ** 2), 4.749, 3)


@pytest.mark.parametrize("alpha", [0, 1e-6])
def test_regression_squared_loss_nn_l1(reg_nn_train_data, alpha):
    X, y = reg_nn_train_data
    reg = SGDRegressor(loss="squared", penalty="nn", learning_rate="constant",
                       eta0=1e-1, alpha=alpha, random_state=0)

    reg.fit(X, y)
    pred = reg.predict(X)
    np.testing.assert_almost_equal(np.mean((pred - y) ** 2), 0.016, 3)
    assert (reg.coef_ >= 0).all()


def test_regression_squared_loss_nn_l2(reg_nn_train_data):
    X, y = reg_nn_train_data

    reg = SGDRegressor(loss="squared", penalty="nnl2", learning_rate="constant",
                       eta0=1e-1, alpha=1e-4, random_state=0)

    reg.fit(X, y)
    pred = reg.predict(X)
    np.testing.assert_almost_equal(np.mean((pred - y) ** 2), 0.016, 3)
    np.testing.assert_almost_equal(reg.coef_.sum(), 2.131, 3)
    assert (reg.coef_ >= 0).all()


def test_regression_squared_loss_multiple_output(reg_train_data):
    X, y = reg_train_data
    reg = SGDRegressor(loss="squared", penalty="l2", learning_rate="constant",
                       eta0=1e-2, random_state=0, max_iter=10)
    Y = np.zeros((len(y), 2))
    Y[:, 0] = y
    Y[:, 1] = y
    reg.fit(X, Y)
    pred = reg.predict(X)
    np.testing.assert_almost_equal(np.mean((pred - Y) ** 2), 4.397, 3)
