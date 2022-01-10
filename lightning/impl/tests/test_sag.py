import numpy as np
import pytest
from scipy import sparse

from sklearn.datasets import make_classification

from lightning.impl.base import BaseClassifier
from lightning.impl.dataset_fast import get_dataset
from lightning.classification import SAGClassifier, SAGAClassifier
from lightning.regression import SAGRegressor, SAGARegressor

from lightning.impl.sgd_fast import ModifiedHuber
from lightning.impl.sgd_fast import SmoothHinge
from lightning.impl.sgd_fast import SquaredHinge
from lightning.impl.sgd_fast import Log
from lightning.impl.sgd_fast import SquaredLoss
from lightning.impl.sag import get_auto_step_size
from lightning.impl.tests.utils import check_predict_proba


class L1Penalty(object):

    def __init__(self, l1=1.0):
        self.l1 = l1

    def projection(self, coef, stepsize):
        return np.fmax(coef - stepsize * self.l1, 0) \
                - np.fmax(-coef - stepsize * self.l1, 0)

    def regularization(self, coef):
        return self.l1 * np.sum(np.abs(coef))


class L2Penalty(object):

    def __init__(self, l2=1.0):
        self.l2 = l2

    def projection(self, coef):
        return coef / (1. + self.l2)

    def regularization(self, coef):
        return self.l2 * np.sum(coef**2)


def _fit_sag(X, y, eta, alpha, loss, max_iter, rng):

    if sparse.issparse(X):
        X = X.toarray()

    n_samples, n_features = X.shape
    n_vectors = y.shape[1]
    g = np.empty((n_samples, n_features))
    coef_ = np.zeros((n_vectors, n_features))

    # Initialize gradients
    for i in range(n_samples):
        p = coef_.dot(X[i])
        g[i] = -loss.get_update(p, y[i]) * X[i]
    d = np.sum(g, axis=0)

    # Main loop
    for _ in range(max_iter):
        for _ in range(n_samples):
            i = rng.randint(n_samples - 1)
            p = coef_.dot(X[i])
            gi = -loss.get_update(p, y[i]) * X[i]
            coef_ -= eta * ((gi - g[i] + d) / n_samples + alpha * coef_)
            d += gi - g[i]
            g[i] = gi
    return coef_


def _fit_saga(X, y, eta, alpha, loss, penalty, max_iter, rng):

    if sparse.issparse(X):
        X = X.toarray()

    n_samples, n_features = X.shape
    n_vectors = y.shape[1]
    g = np.empty((n_samples, n_features))
    coef_ = np.zeros((n_vectors, n_features))

    # Initialize gradients
    for i in range(n_samples):
        p = coef_.dot(X[i])
        g[i] = -loss.get_update(p, y[i]) * X[i]
    d = np.sum(g, axis=0)

    # Main loop
    for _ in range(max_iter):
        for _ in range(n_samples):
            i = rng.randint(n_samples - 1)
            p = coef_.dot(X[i])
            gi = -loss.get_update(p, y[i]) * X[i]
            coef_ -= eta * ((gi - g[i] + d / n_samples) + alpha * coef_)
            if penalty is not None:
                coef_ = penalty.projection(coef_, eta)
            d += gi - g[i]
            g[i] = gi
    return coef_


class PySAGClassifier(BaseClassifier):

    def _get_loss(self, loss):
        losses = {
            "modified_huber": ModifiedHuber(),
            "smooth_hinge": SmoothHinge(self.gamma),
            "squared_hinge": SquaredHinge(1.0),
            "log": Log(),
            "squared": SquaredLoss(),
        }
        return losses[loss]

    def _get_penalty(self, penalty):
        if isinstance(penalty, str):
            penalties = {
                "l1": L1Penalty(self.beta),
                "l2": None,  # Updated inside SAG.
            }
            return penalties[penalty]
        else:
            return penalty

    def __init__(self, eta='auto', alpha=1.0, beta=0.0, loss="smooth_hinge",
                 penalty='l2', gamma=1.0, max_iter=100, random_state=None,
                 callback=None):
        self.eta = eta
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.loss = loss
        self.penalty = penalty
        self.max_iter = max_iter
        self.random_state = random_state
        self.rng = self._get_random_state()
        self.callback = callback

        self.is_saga = False

    def fit(self, X, y):
        self._set_label_transformers(y)
        y_trans = np.asfortranarray(self.label_binarizer_.transform(y),
                                    dtype=np.float64)

        if self.eta is None or self.eta == 'auto':
            eta = get_auto_step_size(
                get_dataset(X, order="c"), self.alpha, self.loss, self.is_saga)
        else:
            eta = self.eta

        if self.alpha * eta == 1:
            # to match the beaviour of SAGA
            # in this case SAGA decreases slightly eta
            eta *= 0.9

        loss = self._get_loss(self.loss)
        self.penalty = self._get_penalty(self.penalty)

        if not self.is_saga and self.penalty is not None:
            raise ValueError("PySAGClassifier only accepts l2 penalty. Please "
                             "use `saga=True` or PySAGAClassifier.")

        if self.is_saga:
            self.coef_ = _fit_saga(X, y_trans, eta, self.alpha, loss,
                                   self.penalty, self.max_iter, self.rng)
        else:
            self.coef_ = _fit_sag(X, y_trans, eta, self.alpha, loss,
                                  self.max_iter, self.rng)


class PySAGAClassifier(PySAGClassifier):

    def __init__(self, eta, alpha=1.0, beta=0.0, loss="smooth_hinge",
                 penalty='l2', gamma=1.0, max_iter=100, random_state=None,
                 callback=None):
        super().__init__(
                eta=eta, alpha=alpha, beta=beta, loss=loss, penalty=penalty,
                gamma=gamma, max_iter=max_iter,
                random_state=random_state, callback=callback)
        self.is_saga = True


@pytest.mark.parametrize("l1", [0.1, 0.5, .99, 1., 2.])
def test_l1_prox(l1):
    x = np.ones(5)
    penalty = L1Penalty(l1=l1)
    if l1 <= 1.:
        np.testing.assert_array_equal(penalty.projection(x, stepsize=1.), x - l1)
        np.testing.assert_array_equal(penalty.projection(-x, stepsize=1.), -x + l1)
    else:
        np.testing.assert_array_equal(penalty.projection(x, stepsize=1.), 0)
        np.testing.assert_array_equal(penalty.projection(-x, stepsize=1.), 0)


@pytest.mark.parametrize("clf",
                         [SAGClassifier(eta=1e-3, max_iter=20, verbose=0, random_state=0),
                          SAGAClassifier(eta=1e-3, max_iter=20, verbose=0, random_state=0),
                          PySAGClassifier(eta=1e-3, max_iter=20, random_state=0)])
def test_sag(clf, bin_train_data):
    X_bin, y_bin = bin_train_data
    clf.fit(X_bin, y_bin)
    assert not hasattr(clf, 'predict_proba')
    assert clf.score(X_bin, y_bin) == 1.0
    assert list(clf.classes_) == [-1, 1]


@pytest.mark.parametrize("SAG_",
                         [SAGAClassifier,
                          SAGClassifier,
                          SAGRegressor,
                          SAGARegressor])
def test_sag_dataset(SAG_, bin_train_data):
    # make sure SAG/SAGA accept a Dataset object as argument
    X_bin, y_bin = bin_train_data
    clf1 = SAG_(eta=1e-3, max_iter=20, verbose=0, random_state=0)
    clf2 = SAG_(eta=1e-3, max_iter=20, verbose=0, random_state=0)
    clf1.fit(get_dataset(X_bin, order='C'), y_bin)
    clf2.fit(X_bin, y_bin)
    np.testing.assert_almost_equal(clf1.coef_, clf2.coef_)


def test_sag_score():
    X, y = make_classification(n_samples=1000, random_state=0)

    pysag = PySAGClassifier(eta=1e-3, alpha=0.0, beta=0.0, max_iter=10,
                            random_state=0)
    sag = SAGClassifier(eta=1e-3, alpha=0.0, beta=0.0, max_iter=10,
                        random_state=0)

    pysag.fit(X, y)
    sag.fit(X, y)
    assert pysag.score(X, y) == sag.score(X, y)


def test_sag_proba():
    n_samples = 10
    X, y = make_classification(n_samples=n_samples, random_state=0)
    sag = SAGClassifier(eta=1e-3, alpha=0.0, beta=0.0, max_iter=10,
                        loss='log', random_state=0)
    sag.fit(X, y)
    check_predict_proba(sag, X)


def test_sag_multiclass_classes():
    X, y = make_classification(n_samples=10, random_state=0, n_classes=3,
                               n_informative=4)
    sag = SAGClassifier()
    sag.fit(X, y)
    assert list(sag.classes_) == [0, 1, 2]


def test_no_reg_sag(bin_train_data):
    X_bin, y_bin = bin_train_data
    pysag = PySAGClassifier(eta=1e-3, alpha=0.0, max_iter=10, random_state=0)
    sag = SAGClassifier(eta=1e-3, alpha=0.0, max_iter=10, random_state=0)

    pysag.fit(X_bin, y_bin)
    sag.fit(X_bin, y_bin)
    np.testing.assert_array_almost_equal(pysag.coef_, sag.coef_)


def test_l2_regularized_sag(bin_train_data):
    X_bin, y_bin = bin_train_data
    pysag = PySAGClassifier(eta=1e-3, alpha=1.0, max_iter=10, random_state=0)
    sag = SAGClassifier(eta=1e-3, alpha=1.0, max_iter=10, random_state=0)

    pysag.fit(X_bin, y_bin)
    sag.fit(X_bin, y_bin)
    np.testing.assert_array_almost_equal(pysag.coef_, sag.coef_)


def test_saga_score():
    X, y = make_classification(n_samples=1000, random_state=0)

    pysaga = PySAGAClassifier(eta=1e-3, alpha=0.0, beta=0.0, max_iter=1,
                              penalty=None, random_state=0)
    saga = SAGAClassifier(eta=1e-3, alpha=0.0, beta=0.0, max_iter=1,
                          penalty=None, random_state=0)

    pysaga.fit(X, y)
    saga.fit(X, y)
    assert pysaga.score(X, y) == saga.score(X, y)


def test_enet_regularized_saga(bin_train_data):
    X_sparse = sparse.rand(100, 50, density=.5, random_state=0).tocsr()
    y_sparse = np.random.randint(0, high=2, size=100)

    eta = 1e-3
    for (X, y) in (bin_train_data, (X_sparse, y_sparse)):
        for alpha in np.logspace(-3, 0, 5):
            for beta in np.logspace(-3, 3, 5):
                pysaga = PySAGAClassifier(
                    eta=eta, alpha=alpha, beta=beta,
                    max_iter=5, penalty='l1', random_state=0)
                saga = SAGAClassifier(
                    eta=eta, alpha=alpha, beta=beta, max_iter=5,
                    penalty='l1', random_state=0, tol=1e-24)

                pysaga.fit(X, y)
                saga.fit(X, y)
                np.testing.assert_array_almost_equal(pysaga.coef_, saga.coef_)


def test_l2_regularized_saga(bin_train_data):
    X_bin, y_bin = bin_train_data
    pysaga = PySAGAClassifier(eta=1e-3, alpha=1.0, max_iter=10,
                              penalty=None, random_state=0)
    saga = SAGAClassifier(eta=1e-3, alpha=1.0, max_iter=10,
                          penalty=None, random_state=0)
    pysaga.fit(X_bin, y_bin)
    saga.fit(X_bin, y_bin)
    np.testing.assert_array_almost_equal(pysaga.coef_, saga.coef_)


def test_elastic_saga(bin_train_data):
    X_bin, y_bin = bin_train_data
    ab = [1e-5, 1e-2, 1e-1, 1.]
    for alpha, beta in zip(ab, ab):
        pysaga = PySAGAClassifier(eta=1e-3, alpha=alpha, beta=beta, max_iter=1,
                                  penalty='l1', random_state=0)
        saga = SAGAClassifier(eta=1e-3, alpha=alpha, beta=beta, max_iter=1,
                              penalty='l1', random_state=0, tol=0)
        pysaga.fit(X_bin, y_bin)
        saga.fit(X_bin, y_bin)
        np.testing.assert_array_almost_equal(pysaga.coef_, saga.coef_)


def test_no_reg_saga(bin_train_data):
    # Using no regularisation at all
    X_bin, y_bin = bin_train_data
    pysaga = PySAGAClassifier(eta=1e-3, alpha=0.0, beta=0.0, max_iter=10,
                              penalty=None, random_state=0)
    saga = SAGAClassifier(eta=1e-3, alpha=0.0, beta=0.0, max_iter=10,
                          penalty=None, random_state=0)

    pysaga.fit(X_bin, y_bin)
    saga.fit(X_bin, y_bin)
    np.testing.assert_array_almost_equal(pysaga.coef_, saga.coef_)


@pytest.mark.parametrize("SAG_",
                         [SAGClassifier,
                          PySAGClassifier,
                          SAGAClassifier,
                          PySAGAClassifier])
def test_sag_callback(SAG_, bin_train_data):
    class Callback(object):

        def __init__(self, X, y):
            self.X = X
            self.y = y
            self.obj = []

        def __call__(self, clf):
            clf._finalize_coef()
            y_pred = clf.decision_function(self.X).ravel()
            loss = (np.maximum(1 - self.y * y_pred, 0) ** 2).mean()
            coef = clf.coef_.ravel()
            regul = 0.5 * clf.alpha * np.dot(coef, coef)
            self.obj.append(loss + regul)

    X_bin, y_bin = bin_train_data
    cb = Callback(X_bin, y_bin)
    clf = SAG_(loss="squared_hinge", eta=1e-3, max_iter=20,
               random_state=0, callback=cb)
    clf.fit(X_bin, y_bin)
    # its not a descent method, just check that most of
    # updates are decreasing the objective function
    assert np.mean(np.diff(cb.obj) <= 0) > 0.9


@pytest.mark.parametrize("clf",
                         [SAGClassifier(loss='log', max_iter=20, verbose=0, random_state=0),
                          SAGAClassifier(loss='log', max_iter=20, verbose=0, random_state=0),
                          PySAGClassifier(loss='log', max_iter=20, random_state=0)])
def test_auto_stepsize(clf, bin_train_data):
    X_bin, y_bin = bin_train_data
    clf.fit(X_bin, y_bin)
    assert clf.score(X_bin, y_bin) == 1.0


@pytest.mark.parametrize("SAG_",
                         [SAGRegressor,
                          SAGARegressor])
def test_sag_regression(SAG_, bin_train_data):
    X_bin, y_bin = bin_train_data
    reg = SAG_(random_state=0, loss='squared')
    reg.fit(X_bin, y_bin)
    y_pred = np.sign(reg.predict(X_bin))
    assert np.mean(y_bin == y_pred) == 1.0


@pytest.mark.parametrize("SAG_",
                         [SAGClassifier,
                          SAGAClassifier])
def test_sag_sparse(SAG_):
    # FIX for https://github.com/mblondel/lightning/issues/33
    # check that SAG has the same results with dense
    # and sparse data
    X = sparse.rand(100, 50, density=.5, random_state=0)
    y = np.random.randint(0, high=2, size=100)
    for alpha in np.logspace(-3, 3, 10):
        clf_sparse = SAG_(eta=1., max_iter=1, random_state=0, alpha=alpha)
        clf_sparse.fit(X, y)
        clf_dense = SAG_(eta=1., max_iter=1, random_state=0, alpha=alpha)
        clf_dense.fit(X.toarray(), y)
        assert clf_sparse.score(X, y) == clf_dense.score(X, y)


def test_sag_sample_weights(train_data):
    X, y = train_data
    clf1 = SAGAClassifier(loss='log', max_iter=5, verbose=0, random_state=0)
    clf2 = SAGAClassifier(loss='log', max_iter=5, verbose=0, random_state=0)
    clf1.fit(X, y)
    sample_weights = [1] * y.size
    clf2.fit(X, y, sample_weight=sample_weights)
    np.testing.assert_array_equal(clf1.coef_.ravel(), clf2.coef_.ravel())

    # same thing but for a regression object
    alpha = 1.0
    clf1 = SAGARegressor(loss='squared', alpha=alpha, max_iter=5, random_state=0)
    clf1.fit(X, y)
    sample_weights = [2] * y.size
    # alpha needs to be multiplied accordingly
    clf2 = SAGARegressor(loss='squared', alpha=2 * alpha, max_iter=5, random_state=0)
    clf2.fit(X, y, sample_weight=sample_weights)
    np.testing.assert_array_equal(clf1.coef_.ravel(), clf2.coef_.ravel())

    #
    # check that samples with a zero weight do not have an influence on the
    # resulting coefficients by adding noise to original samples
    X2 = np.concatenate((X, np.random.randn(*X.shape)), axis=0)   # augment with noise
    y2 = np.concatenate((y, y), axis=0)
    sample_weights = np.ones(y2.size, dtype=float)
    sample_weights[X.shape[0]:] = 0.

    clf1 = SAGARegressor(loss='squared', alpha=alpha,  max_iter=100, random_state=0, tol=1e-24)
    clf1.fit(X, y)
    clf2 = SAGARegressor(loss='squared', alpha=0.5*alpha, max_iter=100, random_state=0, tol=1e-24)
    clf2.fit(X2, y2, sample_weight=sample_weights)
    np.testing.assert_array_almost_equal(clf1.coef_.ravel(), clf2.coef_.ravel(), decimal=6)


def test_sag_adaptive():
    # Check that the adaptive step size strategy yields the same
    # solution as the non-adaptive"""
    np.random.seed(0)
    X = sparse.rand(100, 10, density=.5, random_state=0).tocsr()
    y = np.random.randint(0, high=2, size=100)
    for alpha in np.logspace(-3, 1, 5):
        clf_adaptive = SAGClassifier(
            eta='line-search', random_state=0, alpha=alpha)
        clf_adaptive.fit(X, y)
        clf = SAGClassifier(
            eta='auto', random_state=0, alpha=alpha)
        clf.fit(X, y)
        np.testing.assert_almost_equal(clf_adaptive.score(X, y), clf.score(X, y), 1)

        clf_adaptive = SAGAClassifier(
            eta='line-search', loss='log', random_state=0, alpha=alpha, max_iter=20)
        clf_adaptive.fit(X, y)
        assert np.isnan(clf_adaptive.coef_.sum()) == False
        clf = SAGAClassifier(
            eta='auto', loss='log', random_state=0, alpha=alpha, max_iter=20)
        clf.fit(X, y)
        np.testing.assert_almost_equal(clf_adaptive.score(X, y), clf.score(X, y), 1)
