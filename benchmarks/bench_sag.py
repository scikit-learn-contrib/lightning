import time

import numpy as np

from sklearn.datasets import fetch_20newsgroups_vectorized
from lightning.classification import SAGClassifier

bunch = fetch_20newsgroups_vectorized(subset="all")
X = bunch.data
y = bunch.target
y[y >= 1] = 1

clf = SAGClassifier(eta=1e-4, alpha=1e-5, tol=1e-3, max_iter=20, verbose=1,
                     random_state=0)
start = time.time()
clf.fit(X, y)

print "Training time", time.time() - start
print "Accuracy", np.mean(clf.predict(X) == y)
print "% non-zero", clf.n_nonzero(percentage=True)
