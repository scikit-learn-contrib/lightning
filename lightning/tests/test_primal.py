import numpy as np
import scipy.sparse as sp

from numpy.testing import assert_array_equal, assert_array_almost_equal, \
                          assert_almost_equal
from nose.tools import assert_raises, assert_true, assert_equal

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
            # Caution: use a dense LinearSVC even on sparse data!
            clf = PrimalClassifier(LinearSVC(), metric=metric,
                                   trim_dictionary=False)
            y_pred = clf.fit(X, bin_target).predict(X)
            assert_true(np.mean(y_pred == bin_target) >= 0.98)
            assert_equal(clf.dictionary_.shape[0], X.shape[0])
            assert_equal(clf.n_support_.shape[0], 1)


def test_primal_fit_multiclass():
    for metric in ("rbf", "linear", "poly"):
        for X in (mult_dense, mult_sparse):
            clf = PrimalClassifier(LinearSVC(), metric=metric,
                                   trim_dictionary=False)
            y_pred = clf.fit(X, mult_target).predict(X)
            assert_true(np.mean(y_pred == mult_target) >= 0.8)
            assert_equal(clf.n_support_.shape[0], 3)


def test_primal_fit_proportion():
    clf = PrimalClassifier(LinearSVC(), dictionary_size=0.3)
    clf.fit(bin_dense, bin_target)
    assert_true(clf.coef_.shape, (1, bin_dense.shape[0] / 3))


def test_primal_fit_binary_trim():
    for X in (bin_dense, bin_sparse):
        # Caution: use a dense LinearSVC even on sparse data!
        clf = PrimalClassifier(LinearSVC(penalty="l1", dual=False),
                               trim_dictionary=True)
        y_pred = clf.fit(X, bin_target).predict(X)
        assert_true(np.mean(y_pred == bin_target) >= 0.98)
        assert_true(clf.dictionary_.shape[0] < X.shape[0])


def test_primal_debiasing():
    clf = PrimalClassifier(LinearSVC(penalty="l1", dual=False), debiasing=True)
    clf.fit(bin_dense, bin_target)
    assert_true(density(clf.coef_), 1.0)
    y_pred = clf.predict(bin_dense)
    assert_true(np.mean(y_pred == bin_target) >= 0.98)


def test_primal_coef_():
    n_samples = mult_dense.shape[0]
    n_classes = np.unique(mult_target).shape[0]
    clf = PrimalClassifier(LinearSVC())
    clf.fit(mult_dense, mult_target)
    assert_equal(clf.coef_.shape, (n_classes, n_samples))

def test_primal_cv():
    clf = PrimalClassifierCV(LinearSVC(penalty="l1", dual=False),
                             params=[0.01, 0.1, 1.0],
                             metric="poly",
                             trim_dictionary=False)
    y_pred = clf.fit(mult_dense, mult_target).predict(mult_dense)
