import numpy as np
import pytest

from lightning.classification import AdaGradClassifier
from lightning.regression import AdaGradRegressor
from lightning.impl.adagrad_fast import _proj_elastic_all
from lightning.impl.tests.utils import check_predict_proba


def test_adagrad_elastic_hinge(bin_train_data):
    X_bin, y_bin = bin_train_data
    clf = AdaGradClassifier(alpha=0.5, l1_ratio=0.85, n_iter=10, random_state=0)
    clf.fit(X_bin, y_bin)
    assert not hasattr(clf, "predict_proba")
    assert clf.score(X_bin, y_bin) == 1.0


def test_adagrad_elastic_smooth_hinge(bin_train_data):
    X_bin, y_bin = bin_train_data
    clf = AdaGradClassifier(alpha=0.5, l1_ratio=0.85, loss="smooth_hinge",
                            n_iter=10, random_state=0)
    clf.fit(X_bin, y_bin)
    assert not hasattr(clf, "predict_proba")
    assert clf.score(X_bin, y_bin) == 1.0


def test_adagrad_elastic_log(bin_train_data):
    X_bin, y_bin = bin_train_data
    clf = AdaGradClassifier(alpha=0.1, l1_ratio=0.85, loss="log", n_iter=10,
                            random_state=0)
    clf.fit(X_bin, y_bin)
    assert clf.score(X_bin, y_bin) == 1.0
    check_predict_proba(clf, X_bin)


def test_adagrad_hinge_multiclass(train_data):
    X, y = train_data
    clf = AdaGradClassifier(alpha=1e-2, n_iter=100, loss="hinge", random_state=0)
    clf.fit(X, y)
    assert not hasattr(clf, "predict_proba")
    np.testing.assert_almost_equal(clf.score(X, y), 0.940, 3)


def test_adagrad_classes_binary(bin_train_data):
    X_bin, y_bin = bin_train_data
    clf = AdaGradClassifier()
    assert not hasattr(clf, 'classes_')
    clf.fit(X_bin, y_bin)
    assert list(clf.classes_) == [-1, 1]


def test_adagrad_classes_multiclass(train_data):
    X, y = train_data
    clf = AdaGradClassifier()
    assert not hasattr(clf, 'classes_')
    clf.fit(X, y)
    assert list(clf.classes_) == [0, 1, 2]


def test_adagrad_callback(bin_train_data):
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

    X_bin, y_bin = bin_train_data
    cb = Callback(X_bin, y_bin)
    clf = AdaGradClassifier(alpha=0.5, l1_ratio=0.85, n_iter=10,
                            callback=cb, random_state=0)
    clf.fit(X_bin, y_bin)
    assert cb.acc[-1] == 1.0


@pytest.mark.parametrize("loss", ["squared", "absolute"])
def test_adagrad_regression(loss, bin_train_data):
    X_bin, y_bin = bin_train_data
    reg = AdaGradRegressor(loss=loss)
    reg.fit(X_bin, y_bin)
    y_pred = np.sign(reg.predict(X_bin))
    assert np.mean(y_bin == y_pred) == 1.0
