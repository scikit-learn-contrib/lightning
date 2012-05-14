import numpy as np
import scipy.sparse as sp

from numpy.testing import assert_array_equal, assert_array_almost_equal, \
                          assert_almost_equal
from nose.tools import assert_raises, assert_true, assert_equal, \
                       assert_not_equal

from sklearn.datasets.samples_generator import make_classification

from lightning.primal_kernel import PrimalKernelSVC

bin_dense, bin_target = make_classification(n_samples=200, n_features=100,
                                            n_informative=5,
                                            n_classes=2, random_state=0)


def test_primal_kernel_hinge():
    clf = PrimalKernelSVC(max_iter=1, kernel="rbf", gamma=0.01, C=0.1,
                          random_state=0, loss="hinge")
    clf.fit(bin_dense, bin_target)
    assert_almost_equal(clf.score(bin_dense, bin_target), 0.94)
    assert_equal(clf.n_support_vectors(), 200)


def test_primal_kernel_log():
    clf = PrimalKernelSVC(max_iter=1, kernel="rbf", gamma=0.01, C=0.1,
                          random_state=0, loss="log")
    clf.fit(bin_dense, bin_target)
    assert_almost_equal(clf.score(bin_dense, bin_target), 0.95)
    assert_equal(clf.n_support_vectors(), 200)


def test_primal_kernel_modified_huber():
    clf = PrimalKernelSVC(max_iter=1, kernel="rbf", gamma=0.01, C=0.1,
                          random_state=0, loss="modified_huber")
    clf.fit(bin_dense, bin_target)
    assert_almost_equal(clf.score(bin_dense, bin_target), 0.94)
    assert_equal(clf.n_support_vectors(), 200)
