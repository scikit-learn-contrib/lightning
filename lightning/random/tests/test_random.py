
import numpy as np
from numpy.testing import assert_almost_equal

from lightning.random import RandomState

def test_randint():
    rs = RandomState(seed=0)
    vals = [rs.randint(10) for t in xrange(10000)]
    assert_almost_equal(np.mean(vals), 5.018)

