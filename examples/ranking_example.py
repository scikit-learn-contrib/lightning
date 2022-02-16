"""
================
Ranking example
================
"""

import numpy as np

from sklearn.datasets import load_diabetes
from sklearn.utils.testing import assert_almost_equal

from lightning.ranking import PRank
from lightning.ranking import KernelPRank

# TODO: Add more datasets.
bunch = load_diabetes()
X, y = bunch.data, bunch.target
y = np.round(y, decimals=-2)

# TODO: Add more rankers.
rankers = (PRank(n_iter=10, shuffle=False, random_state=0),
           KernelPRank(kernel="linear",
	               n_iter=10,
                       shuffle=True,
                       random_state=0))

for ranker in rankers:
	print ranker.__class__.__name__
	ranker.fit(X, y)
	print ranker.score(X, y)
