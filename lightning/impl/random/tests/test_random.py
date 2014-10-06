import pickle
import numpy as np
from numpy.testing import (assert_almost_equal, assert_array_equal,
                           assert_equal)
from lightning.impl.random import RandomState


def test_randint():
    rs = RandomState(seed=0)
    vals = [rs.randint(10) for t in xrange(10000)]
    assert_almost_equal(np.mean(vals), 5.018)


def test_shuffle():
    ind = np.arange(10)
    rs = RandomState(seed=0)
    rs.shuffle(ind)
    assert_array_equal(ind, [2, 8, 4, 9, 1, 6, 7, 3, 0, 5])


def test_random_state_pickle():
    rs = RandomState(seed=0)
    random_integer = rs.randint(5)
    pickle_rs = pickle.dumps(rs)
    pickle_rs = pickle.loads(pickle_rs)
    pickle_random_integer = pickle_rs.randint(5)
    assert_equal(random_integer, pickle_random_integer)
