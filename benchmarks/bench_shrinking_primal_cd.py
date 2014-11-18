import time

import numpy as np

from sklearn.datasets import fetch_20newsgroups_vectorized
from lightning.classification import CDClassifier

# Load News20 dataset from scikit-learn.
bunch = fetch_20newsgroups_vectorized(subset="all")
X = bunch.data
y = bunch.target
y[y >= 1] = 1

for shrinking in (True, False):
    clf = CDClassifier(C=1.0, loss="squared_hinge", penalty="l1", tol=1e-3,
                       max_iter=1000, shrinking=shrinking, random_state=0)
    start = time.time()
    clf.fit(X, y)
    print "Training time", time.time() - start
    print "Accuracy", clf.score(X, y)
