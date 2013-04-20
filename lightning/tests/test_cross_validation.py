import numpy as np
import scipy.sparse as sp

from sklearn.utils.testing import assert_equal

from lightning.cross_validation import SparseShuffleSplit

Y = [[0], [1], [0], [2], [3]]
Y = np.array(Y)
Ycsc = sp.csc_matrix(Y)


def test_sparse_shuffle_split():
    cv = SparseShuffleSplit(Ycsc, n_iter=5, test_size=0.3, random_state=0)
    for ind_train, val_train, ind_test, val_test in cv:
        # Check that lengths are correct.
        assert_equal(len(ind_train), len(val_train))
        assert_equal(len(ind_train), 2)
        assert_equal(len(ind_test), len(val_test))
        assert_equal(len(ind_test), 1)

        for i in xrange(len(ind_train)):
            assert_equal(Y[ind_train[i], 0], val_train[i])

        for i in xrange(len(ind_test)):
            assert_equal(Y[ind_test[i], 0], val_test[i])

