
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal

from lightning.random import RandomState


def test_randint():
    rs = RandomState(seed=0)
    vals = [rs.randint(10) for t in xrange(10000)]
    assert_almost_equal(np.mean(vals), 5.018)


def test_shuffle():
    ind = np.arange(10)
    rs = RandomState(seed=0)
    rs.shuffle(ind)
    assert_array_equal(ind, [2, 8, 4, 9, 1, 6, 7, 3, 0, 5])
