import numpy as np
import scipy.sparse as sp

from numpy.testing import assert_array_equal, assert_array_almost_equal, \
                          assert_almost_equal
from nose.tools import assert_raises, assert_true, assert_equal

from sklearn.datasets.samples_generator import make_classification
from sklearn.svm import LinearSVC

from lightning.dual_cd import DualLinearSVC, DualSVC

bin_dense, bin_target = make_classification(n_samples=200, n_features=100,
                                            n_informative=5,
                                            n_classes=2, random_state=0)
bin_sparse = sp.csr_matrix(bin_dense)

mult_dense, mult_target = make_classification(n_samples=300, n_features=100,
                                              n_informative=5,
                                              n_classes=3, random_state=0)
mult_sparse = sp.csr_matrix(mult_dense)


def test_fit_linear_binary():
    clf = DualLinearSVC(loss="l1", random_state=0)
    clf.fit(bin_dense, bin_target)
    y_pred = clf.decision_function(bin_dense)

    clf2 = DualSVC(loss="l1", random_state=0)
    clf2.fit(bin_dense, bin_target)
    y_pred2 = clf2.decision_function(bin_dense)

    assert_array_almost_equal(y_pred, y_pred2)

