import numpy as np
import scipy.sparse as sp

from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_raises
from sklearn.utils.testing import assert_true
from sklearn.utils.testing import assert_false
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_not_equal
from sklearn.utils.testing import assert_greater

from sklearn.datasets.samples_generator import make_classification
from sklearn.datasets.samples_generator import make_regression

from lightning.datasets.samples_generator import make_nn_regression
from lightning.sgd import SGDClassifier
from lightning.sgd import SGDRegressor
from lightning.sgd_fast import Hinge
from lightning.sgd_fast import Log


bin_dense, bin_target = make_classification(n_samples=200, n_features=100,
                                            n_informative=5,
                                            n_classes=2, random_state=0)

mult_dense, mult_target = make_classification(n_samples=300, n_features=100,
                                              n_informative=5,
                                              n_classes=3, random_state=0)

bin_csr = sp.csr_matrix(bin_dense)
mult_csr = sp.csr_matrix(mult_dense)

def test_binary_linear_sgd():
    for data in (bin_dense, bin_csr):
        for clf in (SGDClassifier(random_state=0, loss="hinge",
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
                                  fit_intercept=True, learning_rate="constant"),
                    ):

            clf.fit(data, bin_target)
            assert_greater(clf.score(data, bin_target), 0.934)


def test_multiclass_sgd():
    clf = SGDClassifier(random_state=0)
    clf.fit(mult_dense, mult_target)
    assert_greater(clf.score(mult_dense, mult_target), 0.80)


def test_multiclass_hinge_sgd():
    for data in (mult_dense, mult_csr):
        for fit_intercept in (True, False):
            clf = SGDClassifier(loss="hinge", multiclass=True,
                                fit_intercept=fit_intercept, random_state=0)
            clf.fit(data, mult_target)
            assert_greater(clf.score(data, mult_target), 0.78)


def test_multiclass_hinge_sgd_l1l2():
    for data in (mult_dense, mult_csr):
        clf = SGDClassifier(loss="hinge", penalty="l1/l2",
                            multiclass=True, random_state=0)
        clf.fit(data, mult_target)
        assert_greater(clf.score(data, mult_target), 0.75)


def test_multiclass_squared_hinge_sgd():
    for data in (mult_dense, mult_csr):
        for fit_intercept in (True, False):
            clf = SGDClassifier(loss="squared_hinge", multiclass=True,
                                learning_rate="constant", eta0=1e-3,
                                fit_intercept=fit_intercept, random_state=0)
            clf.fit(data, mult_target)
            assert_greater(clf.score(data, mult_target), 0.78)


def test_multiclass_log_sgd():
    for data in (mult_dense, mult_csr):
        for fit_intercept in (True, False):
            clf = SGDClassifier(loss="log", multiclass=True,
                                fit_intercept=fit_intercept,
                                random_state=0)
            clf.fit(data, mult_target)
            assert_greater(clf.score(data, mult_target), 0.78)


def test_regression_squared_loss():
    X, y = make_regression(n_samples=100, n_features=10, n_informative=8,
                           random_state=0)
    reg = SGDRegressor(loss="squared", penalty="l2", learning_rate="constant",
                       eta0=1e-2, random_state=0)

    reg.fit(X, y)
    pred = reg.predict(X)
    assert_almost_equal(np.mean((pred - y) ** 2), 4.913, 3)


def test_regression_squared_loss_nn_l1():
    X, y, _ = make_nn_regression(n_samples=100, n_features=10, n_informative=8,
                                 random_state=0)

    for alpha in (0, 1e-6):
        reg = SGDRegressor(loss="squared", penalty="nn", learning_rate="constant",
                           eta0=1e-1, alpha=alpha, random_state=0)

        reg.fit(X, y)
        pred = reg.predict(X)
        assert_almost_equal(np.mean((pred - y) ** 2), 0.033, 3)
        assert_false((reg.coef_ < 0).any())


def test_regression_squared_loss_nn_l2():
    X, y, _ = make_nn_regression(n_samples=100, n_features=10, n_informative=8,
                                 random_state=0)

    reg = SGDRegressor(loss="squared", penalty="nnl2", learning_rate="constant",
                       eta0=1e-1, alpha=1e-4, random_state=0)

    reg.fit(X, y)
    pred = reg.predict(X)
    assert_almost_equal(np.mean((pred - y) ** 2), 0.033, 3)
    assert_almost_equal(reg.coef_.sum(), 2.131, 3)
    assert_false((reg.coef_ < 0).any())


def test_regression_squared_loss_multiple_output():
    X, y = make_regression(n_samples=100, n_features=10, n_informative=8,
                           random_state=0)
    reg = SGDRegressor(loss="squared", penalty="l2", learning_rate="constant",
                       eta0=1e-2, random_state=0, max_iter=10)
    Y = np.zeros((len(y), 2))
    Y[:, 0] = y
    Y[:, 1] = y
    reg.fit(X, Y)
    pred = reg.predict(X)
    assert_almost_equal(np.mean((pred - Y) ** 2), 4.541, 3)
