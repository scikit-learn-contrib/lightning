import numpy as np
import scipy.sparse as sp

from numpy.testing import assert_array_equal, assert_array_almost_equal, \
                          assert_almost_equal
from nose.tools import assert_raises, assert_true, assert_equal

from sklearn.datasets.samples_generator import make_classification

from lightning.kmp import KernelMatchingPursuit

bin_dense, bin_target = make_classification(n_samples=200, n_features=100,
                                            n_informative=5,
                                            n_classes=2, random_state=0)
bin_sparse = sp.csr_matrix(bin_dense)

mult_dense, mult_target = make_classification(n_samples=300, n_features=100,
                                              n_informative=5,
                                              n_classes=3, random_state=0)
mult_sparse = sp.csr_matrix(mult_dense)


def test_kmp_fit_binary():
    for metric in ("rbf", "linear", "poly"):
        kmp = KernelMatchingPursuit(n_nonzero_coefs=0.5,
                                    dictionary_size=0.5,
                                    metric=metric,
                                    random_state=0)
        kmp.fit(bin_dense, bin_target)
        assert_equal(kmp.dictionary_.shape[1], bin_dense.shape[0] / 2)
        y_pred = kmp.predict(bin_dense)
        print np.mean(bin_target == y_pred)
