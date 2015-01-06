import numpy as np

from sklearn.datasets import load_iris
from sklearn.preprocessing import Normalizer
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_almost_equal

from lightning.classification import AdaGradClassifier
from lightning.regression import AdaGradRegressor
from lightning.impl.adagrad_fast import _proj_elastic_all

iris = load_iris()
X, y = iris.data, iris.target

X_bin = X[y <= 1]
y_bin = y[y <= 1] * 2 - 1


def test_adagrad_elastic_hinge():
    clf = AdaGradClassifier(alpha=0.5, l1_ratio=0.85, n_iter=10, random_state=0)
    clf.fit(X_bin, y_bin)
    assert_equal(clf.score(X_bin, y_bin), 1.0)


def test_adagrad_elastic_smooth_hinge():
    clf = AdaGradClassifier(alpha=0.5, l1_ratio=0.85, loss="smooth_hinge",
                            n_iter=10, random_state=0)
    clf.fit(X_bin, y_bin)
    assert_equal(clf.score(X_bin, y_bin), 1.0)


def test_adagrad_elastic_log():
    clf = AdaGradClassifier(alpha=0.1, l1_ratio=0.85, loss="log", n_iter=10,
                            random_state=0)
    clf.fit(X_bin, y_bin)
    assert_equal(clf.score(X_bin, y_bin), 1.0)


def test_adagrad_hinge_multiclass():
    clf = AdaGradClassifier(alpha=1e-2, n_iter=100, loss="hinge", random_state=0)
    clf.fit(X, y)
    assert_almost_equal(clf.score(X, y), 0.960, 3)


def test_adagrad_callback():
    class Callback(object):

        def __init__(self, X, y):
            self.X = X
            self.y = y
            self.acc = []

        def __call__(self, clf, t):
            alpha1 = clf.l1_ratio * clf.alpha
            alpha2 = (1 - clf.l1_ratio) * clf.alpha
            _proj_elastic_all(clf.eta, t, clf.g_sum_[0], clf.g_norms_[0],
                              alpha1, alpha2, 0, clf.coef_[0])
            score = clf.score(self.X, self.y)
            self.acc.append(score)

    cb = Callback(X_bin, y_bin)
    clf = AdaGradClassifier(alpha=0.5, l1_ratio=0.85, n_iter=10,
                            callback=cb, random_state=0)
    clf.fit(X_bin, y_bin)
    assert_equal(cb.acc[-1], 1.0)


def test_adagrad_regression():
    for loss in ("squared", "absolute"):
        reg = AdaGradRegressor(loss=loss)
        reg.fit(X_bin, y_bin)
        y_pred = np.sign(reg.predict(X_bin))
        assert_equal(np.mean(y_bin == y_pred), 1.0)
