import time

import numpy as np

from sklearn.datasets import fetch_20newsgroups_vectorized
from lightning.classification import CDClassifier

bunch = fetch_20newsgroups_vectorized(subset="all")
X = bunch.data
y = bunch.target
y[y >= 1] = 1

Cs = np.logspace(-3, 3, 20)

for warm_start in (True, False):
    clf = CDClassifier(loss="squared_hinge", tol=1e-3, max_iter=100,
                       warm_start=warm_start)

    scores = []
    start = time.time()
    for C in Cs:
        clf.C = C
        clf.fit(X, y)
        scores.append(clf.score(X, y))

    print "Total time", time.time() - start
    print "Average accuracy", np.mean(scores)
