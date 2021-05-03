import numpy as np

from sklearn.datasets import load_iris

from lightning.classification import SDCAClassifier
from lightning.regression import SDCARegressor
from lightning.impl.tests.utils import check_predict_proba


iris = load_iris()
X, y = iris.data, iris.target

X_bin = X[y <= 1]
y_bin = y[y <= 1] * 2 - 1


def test_sdca_hinge():
    clf = SDCAClassifier(loss="hinge", random_state=0)
    clf.fit(X_bin, y_bin)
    assert not hasattr(clf, 'predict_proba')
    assert clf.score(X_bin, y_bin) == 1.0


def test_sdca_hinge_multiclass():
    clf = SDCAClassifier(alpha=1e-2, max_iter=100, loss="hinge",
                              random_state=0)
    clf.fit(X, y)
    np.testing.assert_almost_equal(clf.score(X, y), 0.947, 3)


def test_sdca_squared():
    clf = SDCAClassifier(loss="squared", random_state=0)
    clf.fit(X_bin, y_bin)
    assert not hasattr(clf, 'predict_proba')
    assert clf.score(X_bin, y_bin) == 1.0


def test_sdca_absolute():
    clf = SDCAClassifier(loss="absolute", random_state=0)
    clf.fit(X_bin, y_bin)
    assert not hasattr(clf, 'predict_proba')
    assert clf.score(X_bin, y_bin) == 1.0


def test_sdca_hinge_elastic():
    clf = SDCAClassifier(alpha=0.5, l1_ratio=0.85, loss="hinge",
                              random_state=0)
    clf.fit(X_bin, y_bin)
    assert clf.score(X_bin, y_bin) == 1.0


def test_sdca_smooth_hinge_elastic():
    clf = SDCAClassifier(alpha=0.5, l1_ratio=0.85, loss="smooth_hinge",
                              random_state=0)
    clf.fit(X_bin, y_bin)
    assert not hasattr(clf, 'predict_proba')
    assert clf.score(X_bin, y_bin) == 1.0


def test_sdca_squared_hinge_elastic():
    clf = SDCAClassifier(alpha=0.5, l1_ratio=0.85, loss="squared_hinge",
                              random_state=0)
    clf.fit(X_bin, y_bin)
    assert clf.score(X_bin, y_bin) == 1.0


def test_sdca_hinge_l1_only():
    clf = SDCAClassifier(alpha=0.5, l1_ratio=1.0, loss="hinge", tol=1e-2,
                              max_iter=200, random_state=0)
    clf.fit(X_bin, y_bin)
    assert clf.score(X_bin, y_bin) == 1.0


def test_sdca_smooth_hinge_l1_only():
    clf = SDCAClassifier(alpha=0.5, l1_ratio=1.0, loss="smooth_hinge",
                              tol=1e-2, max_iter=200, random_state=0)
    clf.fit(X_bin, y_bin)
    assert clf.score(X_bin, y_bin) == 1.0


def test_sdca_squared_l1_only():
    clf = SDCAClassifier(alpha=0.5, l1_ratio=1.0, loss="squared", tol=1e-2,
                              max_iter=100, random_state=0)
    clf.fit(X_bin, y_bin)
    assert clf.score(X_bin, y_bin) == 1.0


def test_sdca_absolute_l1_only():
    clf = SDCAClassifier(alpha=0.5, l1_ratio=1.0, loss="absolute",
                              tol=1e-2, max_iter=200, random_state=0)
    clf.fit(X_bin, y_bin)
    assert clf.score(X_bin, y_bin) == 1.0


def test_sdca_callback():
    class Callback(object):

        def __init__(self, X, y):
            self.X = X
            self.y = y
            self.acc = []

        def __call__(self, clf):
            score = clf.score(self.X, self.y)
            self.acc.append(score)

    cb = Callback(X_bin, y_bin)
    clf = SDCAClassifier(alpha=0.5, l1_ratio=0.85, loss="hinge",
                              callback=cb, random_state=0)
    clf.fit(X_bin, y_bin)
    assert cb.acc[0] == 0.5
    assert cb.acc[-1] == 1.0


def test_bin_classes():
    clf = SDCAClassifier()
    clf.fit(X_bin, y_bin)
    assert list(clf.classes_) == [-1, 1]


def test_multiclass_classes():
    clf = SDCAClassifier()
    clf.fit(X, y)
    assert list(clf.classes_) == [0, 1, 2]


def test_sdca_regression():
    for loss in ("squared", "absolute"):
        reg = SDCARegressor(loss=loss)
        reg.fit(X_bin, y_bin)
        y_pred = np.sign(reg.predict(X_bin))
        assert np.mean(y_bin == y_pred) == 1.0

