import numpy as np
import pytest

from lightning.classification import SDCAClassifier
from lightning.regression import SDCARegressor


def test_sdca_hinge(bin_train_data):
    X_bin, y_bin = bin_train_data
    clf = SDCAClassifier(loss="hinge", random_state=0)
    clf.fit(X_bin, y_bin)
    assert not hasattr(clf, 'predict_proba')
    assert clf.score(X_bin, y_bin) == 1.0


def test_sdca_hinge_multiclass(train_data):
    X, y = train_data
    clf = SDCAClassifier(alpha=1e-2, max_iter=100, loss="hinge",
                         random_state=0)
    clf.fit(X, y)
    np.testing.assert_almost_equal(clf.score(X, y), 0.933, 3)


def test_sdca_squared(bin_train_data):
    X_bin, y_bin = bin_train_data
    clf = SDCAClassifier(loss="squared", random_state=0)
    clf.fit(X_bin, y_bin)
    assert not hasattr(clf, 'predict_proba')
    assert clf.score(X_bin, y_bin) == 1.0


def test_sdca_absolute(bin_train_data):
    X_bin, y_bin = bin_train_data
    clf = SDCAClassifier(loss="absolute", random_state=0)
    clf.fit(X_bin, y_bin)
    assert not hasattr(clf, 'predict_proba')
    assert clf.score(X_bin, y_bin) == 1.0


def test_sdca_hinge_elastic(bin_train_data):
    X_bin, y_bin = bin_train_data
    clf = SDCAClassifier(alpha=0.5, l1_ratio=0.85, loss="hinge",
                         random_state=0)
    clf.fit(X_bin, y_bin)
    assert clf.score(X_bin, y_bin) == 1.0


def test_sdca_smooth_hinge_elastic(bin_train_data):
    X_bin, y_bin = bin_train_data
    clf = SDCAClassifier(alpha=0.5, l1_ratio=0.85, loss="smooth_hinge",
                         random_state=0)
    clf.fit(X_bin, y_bin)
    assert not hasattr(clf, 'predict_proba')
    assert clf.score(X_bin, y_bin) == 1.0


def test_sdca_squared_hinge_elastic(bin_train_data):
    X_bin, y_bin = bin_train_data
    clf = SDCAClassifier(alpha=0.5, l1_ratio=0.85, loss="squared_hinge",
                         random_state=0)
    clf.fit(X_bin, y_bin)
    assert clf.score(X_bin, y_bin) == 1.0


def test_sdca_hinge_l1_only(bin_train_data):
    X_bin, y_bin = bin_train_data
    clf = SDCAClassifier(alpha=0.5, l1_ratio=1.0, loss="hinge", tol=1e-2,
                         max_iter=200, random_state=0)
    clf.fit(X_bin, y_bin)
    assert clf.score(X_bin, y_bin) == 1.0


def test_sdca_smooth_hinge_l1_only(bin_train_data):
    X_bin, y_bin = bin_train_data
    clf = SDCAClassifier(alpha=0.5, l1_ratio=1.0, loss="smooth_hinge",
                         tol=1e-2, max_iter=200, random_state=0)
    clf.fit(X_bin, y_bin)
    assert clf.score(X_bin, y_bin) == 1.0


def test_sdca_squared_l1_only(bin_train_data):
    X_bin, y_bin = bin_train_data
    clf = SDCAClassifier(alpha=0.5, l1_ratio=1.0, loss="squared", tol=1e-2,
                         max_iter=100, random_state=0)
    clf.fit(X_bin, y_bin)
    assert clf.score(X_bin, y_bin) == 1.0


def test_sdca_absolute_l1_only(bin_train_data):
    X_bin, y_bin = bin_train_data
    clf = SDCAClassifier(alpha=0.5, l1_ratio=1.0, loss="absolute",
                         tol=1e-2, max_iter=200, random_state=0)
    clf.fit(X_bin, y_bin)
    assert clf.score(X_bin, y_bin) == 1.0


def test_sdca_callback(bin_train_data):
    class Callback(object):

        def __init__(self, X, y):
            self.X = X
            self.y = y
            self.acc = []

        def __call__(self, clf):
            score = clf.score(self.X, self.y)
            self.acc.append(score)

    X_bin, y_bin = bin_train_data
    cb = Callback(X_bin, y_bin)
    clf = SDCAClassifier(alpha=0.5, l1_ratio=0.85, loss="hinge",
                         callback=cb, random_state=0)
    clf.fit(X_bin, y_bin)
    assert cb.acc[0] == 0.5
    assert cb.acc[-1] == 1.0


def test_bin_classes(bin_train_data):
    X_bin, y_bin = bin_train_data
    clf = SDCAClassifier()
    clf.fit(X_bin, y_bin)
    assert list(clf.classes_) == [-1, 1]


def test_multiclass_classes(train_data):
    X, y = train_data
    clf = SDCAClassifier()
    clf.fit(X, y)
    assert list(clf.classes_) == [0, 1, 2]


@pytest.mark.parametrize("loss", ["squared", "absolute"])
def test_sdca_regression(bin_train_data, loss):
    X_bin, y_bin = bin_train_data
    reg = SDCARegressor(loss=loss)
    reg.fit(X_bin, y_bin)
    y_pred = np.sign(reg.predict(X_bin))
    assert np.mean(y_bin == y_pred) == 1.0
