import numpy as np
import scipy.sparse as sp

from numpy.testing import assert_array_equal, assert_array_almost_equal, \
                          assert_almost_equal
from nose.tools import assert_raises, assert_true, assert_equal

from sklearn.datasets.samples_generator import make_classification
from sklearn.svm import LinearSVC
from sklearn.metrics.pairwise import pairwise_kernels

from lightning.primal_cd import PrimalLinearSVC, PrimalSVC

bin_dense, bin_target = make_classification(n_samples=200, n_features=100,
                                            n_informative=5,
                                            n_classes=2, random_state=0)
bin_sparse = sp.csr_matrix(bin_dense)

mult_dense, mult_target = make_classification(n_samples=300, n_features=100,
                                              n_informative=5,
                                              n_classes=3, random_state=0)
mult_sparse = sp.csr_matrix(mult_dense)


def test_fit_linear_binary_l1r():
    clf = PrimalLinearSVC(C=1.0, random_state=0, penalty="l1")
    clf.fit(bin_dense, bin_target)
    acc = clf.score(bin_dense, bin_target)
    assert_almost_equal(acc, 1.0)
    n_nz = np.sum(clf.coef_ != 0)

    clf = PrimalLinearSVC(C=0.1, random_state=0, penalty="l1")
    clf.fit(bin_dense, bin_target)
    acc = clf.score(bin_dense, bin_target)
    assert_almost_equal(acc, 0.97)
    n_nz2 = np.sum(clf.coef_ != 0)

    assert_true(n_nz > n_nz2)


def test_fit_rbf_binary_l1r():
    clf = PrimalSVC(C=0.5, kernel="rbf", gamma=0.1, random_state=0, penalty="l1")
    clf.fit(bin_dense, bin_target)
    acc = clf.score(bin_dense, bin_target)
    assert_almost_equal(acc, 0.845)
    n_nz = np.sum(clf.coef_ != 0)
    assert_equal(n_nz, 160)

    K = pairwise_kernels(bin_dense, metric="rbf", gamma=0.1)
    clf2 = PrimalLinearSVC(C=0.5, random_state=0, penalty="l1")
    clf2.fit(K, bin_target)
    acc = clf2.score(K, bin_target)
    assert_almost_equal(acc, 0.845)
    n_nz = np.sum(clf2.coef_ != 0)
    assert_equal(n_nz, 160)


def test_fit_rbf_binary_l1r_selection():
    for selection in ("loss", "active"):
        clf = PrimalSVC(C=0.5, kernel="rbf", gamma=0.1, random_state=0,
                        penalty="l1", selection=selection)
        clf.fit(bin_dense, bin_target)
        acc = clf.score(bin_dense, bin_target)
        assert_true(acc >= 0.74)
        n_nz = np.sum(clf.coef_ != 0)
        assert_true(n_nz <= 85)


def test_fit_rbf_multi():
    clf = PrimalSVC(penalty="l1", kernel="rbf", gamma=0.1, random_state=0)
    clf.fit(mult_dense, mult_target)
    y_pred = clf.predict(mult_dense)
    acc = np.mean(y_pred == mult_target)
    assert_almost_equal(acc, 1.0)


def test_warm_start_l1r():
    clf = PrimalLinearSVC(warm_start=True, random_state=0, penalty="l1")

    clf.C = 0.1
    clf.fit(bin_dense, bin_target)
    n_nz = np.sum(clf.coef_ != 0)

    clf.C = 0.2
    clf.fit(bin_dense, bin_target)
    n_nz2 = np.sum(clf.coef_ != 0)

    assert_true(n_nz < n_nz2)


def test_warm_start_l1r_rbf():
    clf = PrimalSVC(warm_start=True, kernel="rbf", gamma=0.1,
                    random_state=0, penalty="l1")

    clf.C = 0.5
    clf.fit(bin_dense, bin_target)
    n_nz = np.sum(clf.coef_ != 0)

    clf.C = 0.6
    clf.fit(bin_dense, bin_target)
    n_nz2 = np.sum(clf.coef_ != 0)

    assert_true(n_nz < n_nz2)


def test_early_stopping_l1r_rbf():
    clf = PrimalSVC(kernel="rbf", gamma=0.1,
                    termination="n_sv", sv_upper_bound=30,
                    random_state=0, penalty="l1")

    clf.fit(bin_dense, bin_target)
    n_nz = np.sum(clf.coef_ != 0)

    assert_equal(n_nz, 30)


def test_fit_linear_binary_l2r():
    clf = PrimalLinearSVC(C=1.0, random_state=0, penalty="l2")
    clf.fit(bin_dense, bin_target)
    acc = clf.score(bin_dense, bin_target)
    assert_almost_equal(acc, 1.0)

    K = pairwise_kernels(bin_dense, metric="rbf", gamma=0.1)
    clf2 = PrimalLinearSVC(C=0.5, random_state=0, penalty="l2")
    clf2.fit(K, bin_target)
    acc = clf2.score(K, bin_target)
    assert_almost_equal(acc, 1.0)
    n_nz = np.sum(clf2.coef_ != 0)
    assert_equal(n_nz, 200)


def test_fit_rbf_binary_l2r():
    clf = PrimalSVC(C=0.5, kernel="rbf", gamma=0.1, random_state=0, penalty="l2")
    clf.fit(bin_dense, bin_target)
    acc = clf.score(bin_dense, bin_target)
    assert_almost_equal(acc, 1.0)
    n_nz = np.sum(clf.coef_ != 0)
    assert_equal(n_nz, 200) # dense solution...


def test_warm_start_l2r():
    clf = PrimalLinearSVC(warm_start=True, random_state=0, penalty="l2")

    clf.C = 0.1
    clf.fit(bin_dense, bin_target)
    assert_almost_equal(clf.score(bin_dense, bin_target), 1.0)

    clf.C = 0.2
    clf.fit(bin_dense, bin_target)
    assert_almost_equal(clf.score(bin_dense, bin_target), 1.0)


def test_warm_start_l2r_rbf():
    clf = PrimalSVC(warm_start=True, kernel="rbf", gamma=0.1,
                    random_state=0, penalty="l2")

    clf.C = 0.1
    clf.fit(bin_dense, bin_target)
    assert_almost_equal(clf.score(bin_dense, bin_target), 1.0)

    clf.C = 0.2
    clf.fit(bin_dense, bin_target)
    assert_almost_equal(clf.score(bin_dense, bin_target), 1.0)


def test_debiasing():
    clf = PrimalSVC(kernel="rbf", gamma=0.1, penalty="l1l2", C=0.5, Cd=1.0,
                    max_iter=10)
    clf.fit(bin_dense, bin_target)
    assert_equal(clf.n_support_vectors(), 160)
    assert_almost_equal(clf.score(bin_dense, bin_target), 0.845)
    pred = clf.decision_function(bin_dense)

    clf = PrimalSVC(kernel="rbf", gamma=0.1, penalty="l1", C=0.5)
    clf.fit(bin_dense, bin_target)
    assert_equal(clf.n_support_vectors(), 160)
    K = pairwise_kernels(bin_dense, clf.support_vectors_, metric="rbf", gamma=0.1)
    clf = PrimalLinearSVC(max_iter=10, C=1.0)
    clf.fit(K, bin_target)
    assert_almost_equal(clf.score(K, bin_target), 0.845)
    pred2 = clf.decision_function(K)

    assert_array_almost_equal(pred, pred2)

def test_debiasing_warm_start():
    clf = PrimalSVC(kernel="rbf", gamma=0.1, penalty="l1", max_iter=10)
    clf.C = 0.5
    clf.fit(bin_dense, bin_target)
    assert_equal(clf.n_support_vectors(), 160)
    assert_almost_equal(clf.score(bin_dense, bin_target), 0.845)

    clf = PrimalSVC(kernel="rbf", gamma=0.1, penalty="l1", max_iter=10)
    clf.C = 0.500001
    clf.fit(bin_dense, bin_target)
    assert_equal(clf.n_support_vectors(), 191)
    assert_almost_equal(clf.score(bin_dense, bin_target), 0.97)

    clf = PrimalSVC(kernel="rbf", gamma=0.1, penalty="l1", max_iter=10,
                    warm_start=True)
    clf.C = 0.5
    clf.fit(bin_dense, bin_target)
    assert_equal(clf.n_support_vectors(), 160)
    assert_almost_equal(clf.score(bin_dense, bin_target), 0.845)

    clf = PrimalSVC(kernel="rbf", gamma=0.1, penalty="l1", max_iter=10,
                    warm_start=True)
    clf.C = 0.500001
    clf.fit(bin_dense, bin_target)
    assert_equal(clf.n_support_vectors(), 191)
    assert_almost_equal(clf.score(bin_dense, bin_target), 0.97)


def test_early_stopping_l2r_rbf():
    clf = PrimalSVC(kernel="rbf", gamma=0.1,
                    termination="n_sv", sv_upper_bound=30,
                    random_state=0, penalty="l2")

    clf.fit(bin_dense, bin_target)
    n_nz = np.sum(clf.coef_ != 0)

    assert_equal(n_nz, 30)


def test_empty_model():
    clf = PrimalSVC(kernel="rbf", gamma=0.1, C=0.1, penalty="l1")
    clf.fit(bin_dense, bin_target)
    assert_equal(clf.n_support_vectors(), 0)
    acc = clf.score(bin_dense, bin_target)
    assert_equal(acc, 0.5)

    clf = PrimalSVC(kernel="rbf", gamma=0.1, C=0.1, penalty="l1l2")
    clf.fit(bin_dense, bin_target)
    assert_equal(clf.n_support_vectors(), 0)
    acc = clf.score(bin_dense, bin_target)
    assert_equal(acc, 0.5)
