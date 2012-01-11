# Author: Mathieu Blondel
# License: BSD
"""
==========================================
Dependency on the regularization parameter
==========================================

"""
print __doc__

import sys
import time

import numpy as np
import pylab as pl

from sklearn.svm.sparse import LinearSVC
from sklearn.linear_model.sparse import SGDClassifier

from lightning.datasets import load_news20

from sklearn.externals.joblib import Memory
from lightning.datasets import get_data_home

memory = Memory(cachedir=get_data_home(), verbose=0)

def l1_norm(w):
    return np.sum(np.abs(w))

def l0_norm(w):
    return np.sum(w !=0)

@memory.cache
def fit_linearsvc(X_train, y_train, C):
    start = time.time()
    clf = LinearSVC(C=C, penalty="l1", dual=False)
    clf.fit(X_train, y_train)
    return clf, time.time() - start

@memory.cache
def fit_sgd(X_train, y_train, alpha):
    start = time.time()
    clf = SGDClassifier(alpha=alpha, penalty="l1")
    clf.fit(X_train, y_train)
    return clf, time.time() - start

X_train, y_train, X_test, y_test = load_news20()

try:
    algorithm = sys.argv[1]
except:
    algorithm = "linearsvc"

if algorithm == "linearsvc":
    #Cs = np.linspace(0.1, 10, 10)
    #Cs = np.linspace(0.1, 1, 10)
    Cs = np.linspace(0.1, 2, 10)
    res = [fit_linearsvc(X_train, y_train, C=C) for C in Cs]
    param = "C"
elif algorithm == "sgd":
    Cs = np.linspace(0.00001, 0.0001, 10)
    res = [fit_sgd(X_train, y_train, alpha=C) for C in Cs]
    param = "alpha"
else:
    raise ValueError("Wrong algorithm name!")

clfs, train_times = zip(*res)

pl.figure()
pl.plot(Cs, [l1_norm(clf.coef_) for clf in clfs])
pl.xlabel(param)
pl.ylabel('L1 norm')
pl.title('Relation between %s and L1 norm' % param)
#pl.legend(loc="lower left")

pl.figure()
pl.plot(Cs, [l0_norm(clf.coef_) for clf in clfs])
pl.xlabel(param)
pl.ylabel('L0 norm')
pl.title('Relation between %s and L0 norm' % param)

pl.figure()
pl.plot(Cs, train_times)
pl.xlabel(param)
pl.ylabel('Training time')
pl.title('Relation between %s and training time' % param)

pl.show()
