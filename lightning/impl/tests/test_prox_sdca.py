from sklearn.datasets import load_iris
from sklearn.utils.testing import assert_equal

from lightning.impl.prox_sdca import ProxSDCA_Classifier

iris = load_iris()
X, y = iris.data, iris.target
X = X[y <= 1]
y = y[y <= 1]
y = y * 2 - 1


def test_sdca_hinge():
    clf = ProxSDCA_Classifier(loss="hinge")
    clf.fit(X, y)
    assert_equal(clf.score(X, y), 1.0)


def test_sdca_squared():
    clf = ProxSDCA_Classifier(loss="squared")
    clf.fit(X, y)
    assert_equal(clf.score(X, y), 1.0)


def test_sdca_absolute():
    clf = ProxSDCA_Classifier(loss="absolute")
    clf.fit(X, y)
    assert_equal(clf.score(X, y), 1.0)


def test_prox_sdca_hinge():
    clf = ProxSDCA_Classifier(alpha=0.5, l1_ratio=0.85, loss="hinge")
    clf.fit(X, y)
    assert_equal(clf.score(X, y), 1.0)
