import numpy as np
from sklearn.utils.testing import assert_almost_equal

from lightning.impl.penalty import project_l1_ball


def test_proj_l1_ball():
    rng = np.random.RandomState(0)
    v = rng.randn(100)
    w = project_l1_ball(v, z=50)
    assert_almost_equal(np.sum(np.abs(w)), 50)
