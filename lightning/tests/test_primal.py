import numpy as np
import scipy.sparse as sp

from numpy.testing import assert_array_equal, assert_array_almost_equal, \
                          assert_almost_equal
from nose.tools import assert_raises, assert_true, assert_equal

from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
from sklearn.datasets.samples_generator import make_classification
from sklearn.utils import check_random_state
from sklearn.utils.extmath import density

from lightning.primal import PrimalClassifier
from lightning.primal import PrimalClassifierCV

bin_dense, bin_target = make_classification(n_samples=200, n_features=100,
                                            n_informative=5,
                                            n_classes=2, random_state=0)
bin_sparse = sp.csr_matrix(bin_dense)

mult_dense, mult_target = make_classification(n_samples=300, n_features=100,
                                              n_informative=5,
                                              n_classes=3, random_state=0)
mult_sparse = sp.csr_matrix(mult_dense)


def test_primal_fit_binary():
    for metric in ("rbf", "linear", "poly"):
        for X in (bin_dense, bin_sparse):
            clf = PrimalClassifier(SGDClassifier(seed=0), metric=metric,
                                   trim_dictionary=False, random_state=0)
            y_pred = clf.fit(X, bin_target).predict(X)
            assert_true(np.mean(y_pred == bin_target) >= 0.95)
            assert_equal(clf.dictionary_.shape[0], X.shape[0])
            assert_equal(clf.n_support_.shape[0], 1)


def test_primal_fit_multiclass():
    for metric in ("rbf", "poly"):
        for X in (mult_dense, mult_sparse):
            clf = PrimalClassifier(SGDClassifier(seed=0), metric=metric,
                                   trim_dictionary=False, random_state=0)
            y_pred = clf.fit(X, mult_target).predict(X)
            assert_true(np.mean(y_pred == mult_target) >= 0.8)
            assert_equal(clf.n_support_.shape[0], 3)


def test_primal_fit_proportion():
    clf = PrimalClassifier(SGDClassifier(seed=0), dictionary_size=0.3,
                           random_state=0)
    clf.fit(bin_dense, bin_target)
    assert_equal(clf.coef_.shape, (1, 60))


def test_primal_fit_binary_trim():
    for X in (bin_dense, bin_sparse):
        clf = PrimalClassifier(SGDClassifier(penalty="l1", seed=0, alpha=0.01),
                               trim_dictionary=True, random_state=0)
        y_pred = clf.fit(X, bin_target).predict(X)
        assert_true(np.mean(y_pred == bin_target) >= 0.94)
        assert_true(clf.dictionary_.shape[0] < X.shape[0])


def test_primal_debiasing():
    clf = PrimalClassifier(SGDClassifier(penalty="l1", seed=0, alpha=0.01),
                           debiasing=True, random_state=0)
    clf.fit(bin_dense, bin_target)
    assert_true(density(clf.coef_), 1.0)
    y_pred = clf.predict(bin_dense)
    assert_true(np.mean(y_pred == bin_target) >= 0.96)


def test_primal_coef_():
    n_samples = mult_dense.shape[0]
    n_classes = np.unique(mult_target).shape[0]
    clf = PrimalClassifier(SGDClassifier(seed=0), random_state=0)
    clf.fit(mult_dense, mult_target)
    assert_equal(clf.coef_.shape, (n_classes, n_samples))

def test_primal_cv_binary():
    clf = LinearSVC(penalty="l1", dual=False)
    clf = PrimalClassifierCV(clf,
                             params=np.linspace(0.01, 10.0, 5),
                             upper_bound=50,
                             metric="linear",
                             random_state=0)
    y_pred = clf.fit(bin_dense, bin_target).predict(bin_dense)
    assert_equal(clf.n_support_[0], 52)

def test_primal_cv_multiclass():
    clf = PrimalClassifierCV(SGDClassifier(penalty="l1", seed=0),
                             params=np.linspace(0.01, 10.0, 10),
                             upper_bound=50,
                             metric="linear",
                             random_state=0)
    y_pred = clf.fit(mult_dense, mult_target).predict(mult_dense)
    assert_equal(int(np.mean(clf.n_support_)), 50)
