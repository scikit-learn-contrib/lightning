import numpy as np
from scipy import sparse

from sklearn.datasets import load_iris
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.testing import assert_true
from sklearn.utils.testing import assert_equal

from lightning.classification import SAGClassifier
from lightning.regression import SAGRegressor

iris = load_iris()
X, y = iris.data, iris.target

X_bin = X[y <= 1]
y_bin = y[y <= 1] * 2 - 1


def test_sag():
    clf = SAGClassifier(eta=1e-3, max_iter=20, verbose=0, random_state=0)
    clf.fit(X_bin, y_bin)
    assert_equal(clf.score(X_bin, y_bin), 1.0)


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
    clf = SAGClassifier(loss="squared_hinge", eta=1e-3, max_iter=20,
                         random_state=0, callback=cb)
    clf.fit(X_bin, y_bin)
    assert_true(np.all(np.diff(cb.obj) <= 0))


def test_sag_regression():
    reg = SAGRegressor()
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
