import time

import numpy as np

from sklearn.datasets import fetch_20newsgroups_vectorized
from lightning.classification import LinearSVC
from lightning.classification import ProxSDCA_Classifier

bunch = fetch_20newsgroups_vectorized(subset="all")
X = bunch.data
y = bunch.target
y[y >= 1] = 1

alpha = 1e-4

clf1 = LinearSVC(loss="squared_hinge", C=1.0 / (alpha * X.shape[0]), tol=1e-3,
                max_iter=20, random_state=0)
clf2 = ProxSDCA_Classifier(loss="squared_hinge", alpha=alpha, tol=1e-6,
                           max_iter=20, random_state=0)


for clf in (clf1, clf2):
    print clf.__class__.__name__
    start = time.time()
    clf.fit(X, y)

    print "Training time", time.time() - start
    print "Accuracy", np.mean(clf.predict(X) == y)
    print
