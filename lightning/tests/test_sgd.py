import numpy as np
import scipy.sparse as sp

from numpy.testing import assert_array_equal, assert_array_almost_equal, \
                          assert_almost_equal
from nose.tools import assert_raises, assert_true, assert_equal, \
                       assert_not_equal
from sklearn.utils.testing import assert_greater

from sklearn.datasets.samples_generator import make_classification
from lightning.sgd import SGDClassifier


bin_dense, bin_target = make_classification(n_samples=200, n_features=100,
                                            n_informative=5,
                                            n_classes=2, random_state=0)

mult_dense, mult_target = make_classification(n_samples=300, n_features=100,
                                              n_informative=5,
                                              n_classes=3, random_state=0)

def test_binary_linear_sgd():
    clf = SGDClassifier(random_state=0)
    clf.fit(bin_dense, bin_target)
    assert_greater(clf.score(bin_dense, bin_target), 0.99)
