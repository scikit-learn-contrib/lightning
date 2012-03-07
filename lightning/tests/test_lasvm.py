import numpy as np
import scipy.sparse as sp

from numpy.testing import assert_array_equal, assert_array_almost_equal, \
                          assert_almost_equal
from nose.tools import assert_raises, assert_true, assert_equal

from sklearn.datasets.samples_generator import make_classification

from lightning.lasvm import LaSVM
from lightning.lasvm_fast import _remove

bin_dense, bin_target = make_classification(n_samples=200, n_features=100,
                                            n_informative=5,
                                            n_classes=2, random_state=0)
bin_sparse = sp.csr_matrix(bin_dense)

mult_dense, mult_target = make_classification(n_samples=300, n_features=100,
                                              n_informative=5,
                                              n_classes=3, random_state=0)
mult_sparse = sp.csr_matrix(mult_dense)


def test_fit_linear_binary():
    for selection, exp in (("permute", 1.0),
                           ("active", 0.99),
                           ("loss", 1.0)):
        clf = LaSVM(random_state=0, max_iter=2, kernel="linear",
                    selection=selection)
        clf.fit(bin_dense, bin_target)
        acc = clf.score(bin_dense, bin_target)
        assert_almost_equal(acc, exp)


def test_fit_rbf_binary():
    for selection in ("permute", "active", "loss"):
        clf = LaSVM(random_state=0, max_iter=2, kernel="rbf",
                    selection=selection)
        clf.fit(bin_dense, bin_target)
        acc = clf.score(bin_dense, bin_target)
        assert_almost_equal(acc, 1.0)


def test_warm_start():
    for selection in ("permute", "active", "loss"):
        clf = LaSVM(random_state=0, max_iter=2, kernel="rbf", warm_start=True,
                    selection=selection)
        clf.C = 0.5
        clf.fit(bin_dense, bin_target)
        acc = clf.score(bin_dense, bin_target)
        assert_almost_equal(acc, 1.0, 1)

        clf.C = 0.6
        clf.fit(bin_dense, bin_target)
        acc = clf.score(bin_dense, bin_target)
        assert_almost_equal(acc, 1.0)


def test_sv_upper_bound():
    clf = LaSVM(random_state=0, max_iter=2, kernel="rbf", finish_step=True,
                sv_upper_bound=30)
    clf.fit(bin_dense, bin_target)
    n_sv = np.sum(clf.dual_coef_ != 0)

