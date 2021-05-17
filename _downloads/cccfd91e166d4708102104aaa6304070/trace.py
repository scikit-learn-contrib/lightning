"""
===========
Trace norm
===========

"""
print(__doc__)
import numpy as np
from scipy.linalg import svd

from sklearn.datasets import fetch_20newsgroups_vectorized
from sklearn.feature_selection import SelectKBest, chi2

from lightning.classification import FistaClassifier

def rank(M, eps=1e-9):
    U, s, V = svd(M, full_matrices=False)
    return np.sum(s > eps)


bunch = fetch_20newsgroups_vectorized(subset="train")
X_train = bunch.data
y_train = bunch.target

# Reduces dimensionality to make the example faster
ch2 = SelectKBest(chi2, k=5000)
X_train = ch2.fit_transform(X_train, y_train)

bunch = fetch_20newsgroups_vectorized(subset="test")
X_test = bunch.data
y_test = bunch.target
X_test = ch2.transform(X_test)

clf = FistaClassifier(C=1.0 / X_train.shape[0],
                      max_iter=200,
                      penalty="trace",
                      multiclass=True)

for alpha in (1e-3, 1e-2, 0.1, 0.2, 0.3):
    print("alpha=", alpha)
    clf.alpha = alpha
    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))
    print(rank(clf.coef_))
