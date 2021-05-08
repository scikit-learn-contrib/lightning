import pickle
import numpy as np

from lightning.impl.randomkit import RandomState


def test_randint():
    rs = RandomState(seed=0)
    vals = [rs.randint(10) for t in range(10000)]
    np.testing.assert_almost_equal(np.mean(vals), 5.018)


def test_shuffle():
    ind = np.arange(10)
    rs = RandomState(seed=0)
    rs.shuffle(ind)
    np.testing.assert_array_equal(ind, [2, 8, 4, 9, 1, 6, 7, 3, 0, 5])


def test_random_state_pickle():
    rs = RandomState(seed=0)
    random_integer = rs.randint(5)
    pickle_rs = pickle.dumps(rs)
    pickle_rs = pickle.loads(pickle_rs)
    pickle_random_integer = pickle_rs.randint(5)
    assert random_integer == pickle_random_integer
