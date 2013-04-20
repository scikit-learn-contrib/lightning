from math import ceil, floor, factorial

import scipy.sparse as sp

from sklearn.utils import check_random_state


class SparseShuffleSplit(object):

    def __init__(self, y, n_iter=10, test_size=0.1,
                 indices=True, random_state=None):
        assert(sp.issparse(y))
        assert(y.shape[1] == 1)
        self.y = y
        self.n_iter = n_iter
        self.test_size = test_size
        self.random_state = random_state

    def __iter__(self):
        rng = check_random_state(self.random_state)
        nz_rows, _ = self.y.nonzero()
        n_test = int(ceil(len(nz_rows) * self.test_size))
        n_train = len(nz_rows) - n_test
        assert(n_test > 0)
        assert(n_train > 0)
        for i in range(self.n_iter):
            permutation = rng.permutation(len(nz_rows))
            itest = permutation[:n_test]
            itrain = permutation[n_test:n_test + n_train]
            ind_test = nz_rows[itest]
            ind_train = nz_rows[itrain]
            val_test = self.y.data[itest]
            val_train = self.y.data[itrain]

            yield ind_train, val_train, ind_test, val_test


    def __len__(self):
        return self.n_iter

