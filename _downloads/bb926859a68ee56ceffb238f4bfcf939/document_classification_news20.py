"""
================================
Classification of text documents
================================

"""
import numpy as np

from sklearn.datasets import fetch_20newsgroups_vectorized

try:
    from sklearn.model_selection import train_test_split
except ImportError:
    from sklearn.cross_validation import train_test_split
    
from lightning.classification import CDClassifier
from lightning.classification import LinearSVC
from lightning.classification import SGDClassifier

# Load News20 dataset from scikit-learn.
bunch = fetch_20newsgroups_vectorized(subset="all")
X = bunch.data
y = bunch.target

# Select a subset of the classes for faster training.
ind = np.arange(X.shape[0])
subset = y < 5
X = X[ind[subset]]
y = y[subset]

# Train / test split.
X_tr, X_te, y_tr, y_te = train_test_split(X, y,
                                          train_size=0.75,
                                          test_size=0.25,
                                          random_state=0)

clfs = (CDClassifier(loss="squared_hinge",
                     penalty="l2",
                     max_iter=20,
                     random_state=0),

        LinearSVC(max_iter=20,
                  random_state=0),

        SGDClassifier(learning_rate="constant",
                      alpha=1e-3,
                      max_iter=20,
                      random_state=0))

for clf in clfs:
    print(clf.__class__.__name__)
    clf.fit(X_tr, y_tr)
    print(clf.score(X_te, y_te))
