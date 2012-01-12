# Author: Mathieu Blondel
# License: BSD
"""
==========================================
Dependency on the regularization parameter
==========================================

C: the smaller, the more regularization.

alpha: the bigger, the more regularization.

nu: An upper bound on the fraction of training errors and a lower
    bound of the fraction of support vectors. Should be in the
    interval (0, 1].

For LinearSVC with L1-penalty (primal objective):
    - strongly regularized models are sparser (in the primal coef)
    - strongly regularized models are faster to train
    - dependency is rougly linear between C and training time/norms

For SVC (dual objective):
    - strongly regularized models are less sparse (in the dual coef): more SV!
    - RBF and linear kernels:
        lightly regularized models are faster to train at first, but when the
        number of SV stops decreasing (reaches it smallest value),
        the training time starts exploding
    - Polynomial kernel:
        training time always decreases with number of support vectors

For NuSVC:
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

from sklearn.svm.sparse import LinearSVC, SVC, NuSVC
from sklearn.linear_model.sparse import SGDClassifier, Lasso
from sklearn.multiclass import OneVsRestClassifier

from lightning.datasets import load_news20, load_usps, load_mnist

from sklearn.externals.joblib import Memory
from lightning.datasets import get_data_home

memory = Memory(cachedir=get_data_home(), verbose=0, compress=6)

def l1_norm(coef):
    return np.sum(np.abs(coef))

def l0_norm(coef):
    return np.sum(coef != 0)

@memory.cache
def fit_linearsvc(X_train, y_train, C):
    start = time.time()
    clf = LinearSVC(C=C, penalty="l1", dual=False)
    clf.fit(X_train, y_train)
    return clf, time.time() - start

@memory.cache
def fit_svc(X_train, y_train, C, kernel, gamma=0.1, degree=4, coef0=1):
    start = time.time()
    clf = SVC(C=C, kernel=kernel, degree=4, coef0=1.0)
    clf.fit(X_train, y_train)
    return clf, time.time() - start

@memory.cache
def fit_nusvc(X_train, y_train, nu, kernel, gamma=0.1, degree=4, coef0=1):
    start = time.time()
    clf = NuSVC(nu=nu, kernel=kernel, degree=4, coef0=1.0)
    clf.fit(X_train, y_train)
    return clf, time.time() - start

@memory.cache
def fit_sgd(X_train, y_train, alpha):
    start = time.time()
    clf = SGDClassifier(alpha=alpha, penalty="l1")
    clf.fit(X_train, y_train)
    return clf, time.time() - start

@memory.cache
def fit_lasso(X_train, y_train, alpha):
    start = time.time()
    base_clf = Lasso(alpha=alpha)
    clf = OneVsRestClassifier(base_clf)
    clf.fit(X_train, y_train)
    return clf, time.time() - start

@memory.cache
def predict(clf, X_test, y_test):
    start = time.time()
    y_pred = clf.predict(X_test)
    return np.mean(y_test == y_pred), time.time() - start

try:
    algorithm = sys.argv[1]
except:
    algorithm = "linearsvc"

try:
    dataset = sys.argv[2]
except:
    dataset = "news20"

loaders = { "news20" : load_news20,
            "usps": load_usps,
            "mnist": load_mnist }
try:
    X_train, y_train, X_test, y_test = loaders[dataset]()
except KeyError:
    raise ValueError("Wrong dataset name!")

if algorithm == "linearsvc":
    #Cs = np.linspace(0.1, 10, 10)
    #Cs = np.linspace(0.1, 1, 10)
    Cs = np.linspace(0.1, 2, 10)
    res = [fit_linearsvc(X_train, y_train, C=C) for C in Cs]
    param = "C"
elif algorithm.startswith("nusvc_"):
    Cs = np.linspace(0.1, 0.5, 10)
    param = "nu"
    kernel = algorithm.replace("nusvc_", "")
    if len(kernel) == 0:
        raise ValueError("Use one of: nusvc_rbf, nusvc_poly, nusvc_linear...")
    res = [fit_nusvc(X_train, y_train, nu=nu, kernel=kernel) for nu in Cs]
elif algorithm.startswith("svc_"):
    Cs = np.linspace(0.1, 2, 10)
    param = "C"
    kernel = algorithm.replace("svc_", "")
    if len(kernel) == 0:
        raise ValueError("Use one of: svc_rbf, svc_poly, svc_linear...")
    res = [fit_svc(X_train, y_train, C=C, kernel=kernel) for C in Cs]
elif algorithm == "sgd":
    Cs = np.linspace(0.00001, 0.0001, 10)
    res = [fit_sgd(X_train, y_train, alpha=C) for C in Cs]
    param = "alpha"
elif algorithm == "lasso":
    Cs = np.linspace(0.1, 1.0, 10)
    res = [fit_lasso(X_train, y_train, alpha=C) for C in Cs]
    param = "alpha"
else:
    raise ValueError("Wrong algorithm name!")

clfs, train_times = zip(*res)

res = [predict(clf, X_test, y_test) for clf in clfs]
accuracies, test_times = zip(*res)

if "svc" in algorithm:
    n_samples = X_train.shape[0]
    pl.figure()
    pl.plot(Cs, [np.mean(1.0 * clf.n_support_ / n_samples) for clf in clfs])
    pl.xlabel(param)
    pl.ylabel('Number of support vectors')
    pl.title('Relation between %s and #nSV' % param)
else:
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

pl.figure()
pl.plot(Cs, accuracies)
pl.xlabel(param)
pl.ylabel('Accuracy')
pl.title('Relation between %s and accuracy' % param)

pl.show()
