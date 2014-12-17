import time

import numpy as np

from sklearn.datasets import fetch_20newsgroups_vectorized
from lightning.classification import FistaClassifier

bunch = fetch_20newsgroups_vectorized(subset="all")
X = bunch.data
y = bunch.target
y[y >= 1] = 1

clf = FistaClassifier(C=1./X.shape[0], alpha=1e-5, max_iter=200)
start = time.time()
clf.fit(X, y)

print "Training time", time.time() - start
print "Accuracy", np.mean(clf.predict(X) == y)
print "% non-zero", clf.n_nonzero(percentage=True)
