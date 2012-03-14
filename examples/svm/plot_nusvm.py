# Author: Mathieu Blondel
# License: BSD
"""
===============================
Dependency on the nu parameter
===============================

nu: An upper bound on the fraction of training errors and a lower
    bound of the fraction of support vectors. Should be in the
    interval (0, 1].

- training time increases linearly with nu
- sparsity increases linearly when nu decreases
- conclusion: one has to increase training error to increase sparsity
- caveat: increasing nu doesn't necessarily improve test accuracy
  (overfitting!)
"""
print __doc__

import sys
import time

import numpy as np
import pylab as pl

from sklearn.svm import NuSVC
from lightning.datasets import load_dataset

from sklearn.externals.joblib import Memory
from lightning.datasets import get_data_home

memory = Memory(cachedir=get_data_home(), verbose=0, compress=6)


@memory.cache
def fit_nusvc(X_train, y_train, nu, kernel, gamma=0.1, degree=4, coef0=1):
    print "Training, nu = ", nu
    start = time.time()
    clf = NuSVC(nu=nu, kernel=kernel, degree=degree, coef0=coef0)
    clf.fit(X_train, y_train)
    return clf, time.time() - start

@memory.cache
def predict(clf, X_test, y_test):
    start = time.time()
    y_pred = clf.predict(X_test)
    return np.mean(y_test == y_pred), time.time() - start

try:
    dataset = sys.argv[1]
except:
    dataset = "usps0"

try:
    kernel = sys.argv[2]
except:
    kernel = "rbf"

try:
    X_train, y_train, X_test, y_test = load_dataset(dataset)
except KeyError:
    raise ValueError("Wrong dataset name!")

Nu = np.linspace(0.01, 0.15, 10)
res = [fit_nusvc(X_train, y_train, nu=nu, kernel=kernel) for nu in Nu]

clfs, train_times = zip(*res)

res = [predict(clf, X_test, y_test) for clf in clfs]
accuracies, test_times = zip(*res)

n_samples = X_train.shape[0]
pl.figure()
pl.plot(Nu, [np.mean(clf.n_support_) for clf in clfs])
pl.xlabel("nu")
pl.ylabel('Number of support vectors')
pl.title('Relation between nu and #nSV')

pl.figure()
pl.plot(Nu, train_times)
pl.xlabel("nu")
pl.ylabel('Training time')
pl.title('Relation between nu and training time')

pl.figure()
pl.plot(Nu, accuracies)
pl.xlabel("nu")
pl.ylabel('Accuracy')
pl.title('Relation between nu and accuracy')

pl.show()
