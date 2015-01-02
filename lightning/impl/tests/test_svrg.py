import numpy as np

from sklearn.datasets import load_iris
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.testing import assert_equal

from lightning.classification import SVRGClassifier
from lightning.impl.sgd_fast import SmoothHinge

iris = load_iris()
X, y = iris.data, iris.target

X_bin = X[y <= 1]
y_bin = y[y <= 1]


def test_svrg():
    clf = SVRGClassifier(eta=1e-3, max_iter=20, random_state=0, verbose=0)
    clf.fit(X_bin, y_bin)
    assert_equal(clf.score(X_bin, y_bin), 1.0)
