import numpy as np
import scipy.sparse as sp

from numpy.testing import assert_array_equal, assert_array_almost_equal, \
                          assert_almost_equal
from nose.tools import assert_raises, assert_true, assert_equal

from sklearn.datasets.samples_generator import make_classification
from sklearn.svm import LinearSVC

from lightning.dual_cd import DualSVC

bin_dense, bin_target = make_classification(n_samples=200, n_features=100,
                                            n_informative=5,
                                            n_classes=2, random_state=0)
bin_csr = sp.csr_matrix(bin_dense)

mult_dense, mult_target = make_classification(n_samples=300, n_features=100,
                                              n_informative=5,
                                              n_classes=3, random_state=0)
mult_sparse = sp.csr_matrix(mult_dense)


def test_fit_linear_binary():
    for data in (bin_dense,):
        for loss in ("l1", "l2"):
            clf = DualSVC(loss=loss, random_state=0, max_iter=100)
            clf.fit(data, bin_target)
            assert_equal(clf.score(data, bin_target), 1.0)


def test_fit_rbf_binary():
    for shrinking in (True, False):
        for selection in ("loss", "cyclic", "active"):
            for loss in ("l1", "l2"):
                clf = DualSVC(loss=loss, kernel="rbf", gamma=0.1, random_state=0,
                              shrinking=shrinking, selection=selection)
                clf.fit(bin_dense, bin_target)
                assert_equal(clf.score(bin_dense, bin_target), 1.0)


def test_fit_rbf_multi():
    clf = DualSVC(kernel="rbf", gamma=0.1, random_state=0)
    clf.fit(mult_dense, mult_target)
    y_pred = clf.predict(mult_dense)
    acc = np.mean(y_pred == mult_target)
    assert_almost_equal(acc, 1.0)


def test_fit_rbf_binary_early_stopping():
    clf = DualSVC(loss="l1", kernel="rbf", gamma=0.5, random_state=0,
                  shrinking=True, selection="loss",
                  termination="n_components", n_components=30)
    clf.fit(bin_dense, bin_target)
    y_pred = clf.predict(bin_dense)
    assert_equal(clf.dual_coef_.shape[1], 30)


#def test_precomputed_kernel():
    #clf = DualSVC(loss="l1", kernel="linear", random_state=0)
    #clf.fit(bin_dense, bin_target)
    #y_pred = clf.decision_function(bin_dense)

    #clf = DualSVC(loss="l1", kernel="precomputed", random_state=0)
    #K = np.dot(bin_dense, bin_dense.T)
    #clf.fit(K, bin_target)
    #y_pred2 = clf.decision_function(K)

    #assert_array_almost_equal(y_pred, y_pred2)


def test_warm_start():
    clf = DualSVC(warm_start=True, loss="l1", kernel="linear", random_state=0,
                  max_iter=100)
    clf2 = DualSVC(warm_start=True, loss="l1", random_state=0,
                         max_iter=100)

    for C in (0.1, 0.2):
        clf.C = C
        clf2.C = C

        clf.fit(bin_dense, bin_target)
        y_pred = clf.decision_function(bin_dense)
        clf2.fit(bin_dense, bin_target)
        y_pred2 = clf2.decision_function(bin_dense)

        assert_array_almost_equal(y_pred, y_pred2)

