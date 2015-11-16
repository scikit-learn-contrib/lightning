import numpy as np
from scipy import sparse

from numpy.testing import assert_array_equal

from sklearn.datasets import load_iris, make_classification
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.testing import assert_true
from sklearn.utils.testing import assert_equal

from lightning.impl.base import BaseClassifier
from lightning.classification import SAGClassifier, SAGAClassifier
from lightning.regression import SAGRegressor, SAGARegressor

from lightning.impl.sgd_fast import ModifiedHuber
from lightning.impl.sgd_fast import SmoothHinge
from lightning.impl.sgd_fast import SquaredHinge
from lightning.impl.sgd_fast import Log
from lightning.impl.sgd_fast import SquaredLoss


iris = load_iris()
X, y = iris.data, iris.target

X_bin = X[y <= 1]
y_bin = y[y <= 1] * 2 - 1


class L1Penalty(object):

    def __init__(self, l1=1.0):
        self.l1 = l1

    def projection(self, coef, stepsize):
        return np.fmax(coef - stepsize * self.l1, 0) \
                -  np.fmax(-coef - stepsize * self.l1, 0)
        
    def regularization(self, coef):
        return self.l1 * np.sum(np.abs(coef))


class L2Penalty(object):

    def __init__(self, l2=1.0):
        self.l2 = l2

    def projection(self, coef):
        return coef / (1. + self.l2)

    def regularization(self, coef):
        return self.l2 * np.sum(coef**2)


def _fit_sag(X, y, eta, alpha, loss, penalty, max_iter, rng):

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
            if penalty is not None:
                coef_ = penalty.projection(coef_, eta)
            d += gi - g[i]
            g[i] = gi
    return coef_


def _fit_saga(X, y, eta, alpha, loss, penalty, max_iter, rng):

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
                "l2": None, # Updated inside SAG.
            }
            return penalties[penalty]
        else:
            return penalty

    def __init__(self, eta, alpha=1.0, beta=0.0, loss="smooth_hinge",
                 penalty='l2', gamma=1.0, max_iter=100, random_state=None,
                 callback=None):
        self.eta = eta
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.loss = self._get_loss(loss)
        self.penalty = self._get_penalty(penalty)
        self.max_iter = max_iter
        self.random_state = random_state
        self.rng = self._get_random_state()
        self.callback = callback

        self.is_saga = False

    def fit(self, X, y):

        if not self.is_saga and self.penalty is not None:
            raise ValueError("PySAGClassifier only accepts l2 penalty. Please "
                             "use `saga=True` or PySAGAClassifier.")

        self.label_binarizer_ = LabelBinarizer(neg_label=-1, pos_label=1)
        y = np.asfortranarray(self.label_binarizer_.fit_transform(y),
                              dtype=np.float64)

        if self.is_saga:
            self.coef_ = _fit_saga(X, y, self.eta, self.alpha, self.loss,
                                   self.penalty, self.max_iter, self.rng)
        else:
            self.coef_ = _fit_sag(X, y, self.eta, self.alpha, self.loss, 
                                  self.penalty, self.max_iter, self.rng)


class PySAGAClassifier(PySAGClassifier):

    def __init__(self, eta, alpha=1.0, beta=0.0, loss="smooth_hinge",
                 penalty='l2', gamma=1.0, max_iter=100, random_state=None,
                 callback=None):
        super(PySAGAClassifier, self).__init__(
                eta=eta, alpha=alpha, beta=beta, loss=loss, penalty=penalty,
                gamma=gamma, max_iter=max_iter,
                random_state=random_state, callback=callback)
        self.is_saga = True


def test_l1_prox():
    x = np.ones(5)
    for l1 in [0.1, 0.5, .99, 1.]:
        penalty = L1Penalty(l1=l1)
        assert_array_equal(penalty.projection(x, stepsize=1.), x - l1)
        assert_array_equal(penalty.projection(-x, stepsize=1.), -x + l1)
    
    penalty = L1Penalty(l1=2.)
    assert_array_equal(penalty.projection(x, stepsize=1.), 0)
    assert_array_equal(penalty.projection(-x, stepsize=1.), 0)
    

def test_sag():
    for clf in (
        SAGClassifier(eta=1e-3, max_iter=20, verbose=0, random_state=0),
        SAGAClassifier(eta=1e-3, max_iter=20, verbose=0, random_state=0),
        PySAGClassifier(eta=1e-3, max_iter=20, random_state=0)
            ):
        clf.fit(X_bin, y_bin)
        assert_equal(clf.score(X_bin, y_bin), 1.0)


def test_sag_score():
    X, y = make_classification(1000, random_state=0)

    pysag = PySAGClassifier(eta=1e-3, alpha=0.0, beta=0.0, max_iter=10,
                            random_state=0)
    sag = SAGClassifier(eta=1e-3, alpha=0.0, beta=0.0, max_iter=10,
                        random_state=0)

    pysag.fit(X, y)
    sag.fit(X, y)
    assert_equal(pysag.score(X, y), sag.score(X, y))


def test_no_reg_sag():
    
    pysag = PySAGClassifier(eta=1e-3, alpha=0.0, max_iter=10, random_state=0)
    sag = SAGClassifier(eta=1e-3, alpha=0.0, max_iter=10, random_state=0)

    pysag.fit(X_bin, y_bin)
    sag.fit(X_bin, y_bin)
    np.testing.assert_array_almost_equal(pysag.coef_, sag.coef_)


def test_l2_regularized_sag():
    
    pysag = PySAGClassifier(eta=1e-3, alpha=1.0, max_iter=10, random_state=0)
    sag = SAGClassifier(eta=1e-3, alpha=1.0, max_iter=10, random_state=0)

    pysag.fit(X_bin, y_bin)
    sag.fit(X_bin, y_bin)
    np.testing.assert_array_almost_equal(pysag.coef_, sag.coef_)


def test_saga_score():
    X, y = make_classification(1000, random_state=0)

    pysaga = PySAGAClassifier(eta=1e-3, alpha=0.0, beta=0.0, max_iter=1,
                              penalty=None, random_state=0)
    saga = SAGAClassifier(eta=1e-3, alpha=0.0, beta=0.0, max_iter=1,
                          penalty=None, random_state=0)

    pysaga.fit(X, y)
    saga.fit(X, y)
    assert_equal(pysaga.score(X, y), saga.score(X, y))


def test_l1_regularized_saga():
    beta = 1e-3
    pysaga = PySAGAClassifier(eta=1e-3, alpha=0.0, beta=beta, max_iter=10,
                              penalty='l1', random_state=0)
    saga = SAGAClassifier(eta=1e-3, alpha=0.0, beta=beta, max_iter=10,
                          penalty='l1', random_state=0)
    pysaga.fit(X_bin, y_bin)
    saga.fit(X_bin, y_bin)
    np.testing.assert_array_almost_equal(pysaga.coef_, saga.coef_)


def test_l2_regularized_saga():
    pysaga = PySAGAClassifier(eta=1e-3, alpha=1.0, max_iter=10,
                              penalty=None, random_state=0)
    saga = SAGAClassifier(eta=1e-3, alpha=1.0, max_iter=10,
                          penalty=None, random_state=0)
    pysaga.fit(X_bin, y_bin)
    saga.fit(X_bin, y_bin)
    np.testing.assert_array_almost_equal(pysaga.coef_, saga.coef_)


def test_no_reg_saga():
    # Using no regularisation at all
    pysaga = PySAGAClassifier(eta=1e-3, alpha=0.0, beta=0.0, max_iter=10,
                              penalty=None, random_state=0)
    saga = SAGAClassifier(eta=1e-3, alpha=0.0, beta=0.0, max_iter=10,
                          penalty=None, random_state=0)

    pysaga.fit(X_bin, y_bin)
    saga.fit(X_bin, y_bin)
    np.testing.assert_array_almost_equal(pysaga.coef_, saga.coef_)


def test_sag_callback():
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

    cb = Callback(X_bin, y_bin)
    for clf in (
        SAGClassifier(loss="squared_hinge", eta=1e-3, max_iter=20,
                      random_state=0, callback=cb),
        PySAGClassifier(loss="squared_hinge", eta=1e-3, max_iter=20,
                        random_state=0, callback=cb),
        SAGAClassifier(loss="squared_hinge", eta=1e-3, max_iter=20,
                       random_state=0, callback=cb),
        PySAGAClassifier(loss="squared_hinge", eta=1e-3, max_iter=20,
                        random_state=0, callback=cb)
            ):
        clf.fit(X_bin, y_bin)
        # its not a descent method, just check that most of
        # updates are decreasing the objective function
        assert_true(np.mean(np.diff(cb.obj) <= 0) > 0.9)


def test_sag_regression():
    for reg in (
        SAGRegressor(random_state=0),
        SAGARegressor(random_state=0, eta=.05)
            ):
        reg.fit(X_bin, y_bin)
        y_pred = np.sign(reg.predict(X_bin))
        assert_equal(np.mean(y_bin == y_pred), 1.0)


def test_sag_sparse():
    # FIX for https://github.com/mblondel/lightning/issues/33
    # check that SAG has the same results with dense
    # and sparse data
    X = sparse.rand(100, 50, density=.5, random_state=0)
    y = np.random.randint(0, high=2, size=100)
    for alpha in np.logspace(-3, 3, 10):
        clf_sparse = SAGClassifier(max_iter=1, random_state=0, alpha=alpha)
        clf_sparse.fit(X, y)
        clf_dense = SAGClassifier(max_iter=1, random_state=0, alpha=alpha)
        clf_dense.fit(X.toarray(), y)
        assert_equal(clf_sparse.score(X, y), clf_dense.score(X, y))

        clf_sparse = SAGAClassifier(max_iter=1, random_state=0, alpha=alpha)
        clf_sparse.fit(X, y)
        clf_dense = SAGAClassifier(max_iter=1, random_state=0, alpha=alpha)
        clf_dense.fit(X.toarray(), y)
        assert_equal(clf_sparse.score(X, y), clf_dense.score(X, y))
