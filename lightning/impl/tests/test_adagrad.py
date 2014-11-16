import numpy as np

from sklearn.datasets import load_iris
from sklearn.preprocessing import Normalizer
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_almost_equal

from lightning.classification import AdaGradClassifier

iris = load_iris()
X, y = iris.data, iris.target

X_bin = X[y <= 1]
y_bin = y[y <= 1]


def test_adagrad_elastic_hinge():
    clf = AdaGradClassifier(alpha=0.5, l1_ratio=0.85, n_iter=10, random_state=0)
    clf.fit(X_bin, y_bin)
    assert_equal(clf.score(X_bin, y_bin), 1.0)


def test_adagrad_elastic_log():
    clf = AdaGradClassifier(alpha=0.5, l1_ratio=0.85, loss="log", n_iter=10,
                            random_state=0)
    clf.fit(X_bin, y_bin)
    assert_equal(clf.score(X_bin, y_bin), 1.0)


def test_adagrad_hinge_multiclass():
    clf = AdaGradClassifier(alpha=1e-2, n_iter=100, loss="hinge", random_state=0)
    clf.fit(X, y)
    assert_almost_equal(clf.score(X, y), 0.953, 3)
