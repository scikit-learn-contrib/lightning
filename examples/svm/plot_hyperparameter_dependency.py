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
from optparse import OptionParser

import numpy as np
import pylab as pl
import matplotlib.font_manager as fm

from lightning.dual_cd import DualSVC
from lightning.primal_cd import PrimalSVC
from lightning.datasets import load_dataset

from sklearn.externals.joblib import Memory
from lightning.datasets import get_data_home

memory = Memory(cachedir=get_data_home(), verbose=0, compress=6)

def set_axes_size(pl):
    ax = pl.axes()
    for xlabel in ax.get_xticklabels():
        xlabel.set_fontsize(13)

    for ylabel in ax.get_yticklabels():
        ylabel.set_fontsize(13)


@memory.cache
def fit_primal_svc(X_train, y_train, C, kernel, gamma=0.1, degree=4, coef0=1):
    print "Training primal SVC, C = ", C
    start = time.time()
    clf = PrimalSVC(C=C, kernel=kernel, degree=degree, coef0=coef0,
                    max_iter=100, tol=1e-4, verbose=1)
    clf.fit(X_train, y_train)
    return clf, time.time() - start


@memory.cache
def fit_dual_svc(X_train, y_train, C, kernel, gamma=0.1, degree=4, coef0=1):
    print "Training dual SVC, C = ", C
    start = time.time()
    clf = DualSVC(C=C, kernel=kernel, degree=degree, coef0=coef0, max_iter=100,
                  loss="l1", tol=1e-4, verbose=1)
    clf.fit(X_train, y_train)
    return clf, time.time() - start


@memory.cache
def predict(clf, X_test, y_test):
    start = time.time()
    y_pred = clf.predict(X_test)
    return np.mean(y_test == y_pred), time.time() - start


op = OptionParser()
op.add_option("--notitle", action="store_true", default=False, dest="notitle")
op.add_option("--bw", action="store_true", default=False, dest="bw")
op.add_option("--kernel", action="store", default="rbf", dest="kernel",
              type="str")
op.add_option("--gamma", action="store", default=0.1, dest="gamma",
              type="float")
op.add_option("--degree", action="store", default=4, dest="degree",
              type="int")

(opts, args) = op.parse_args()

try:
    dataset = args[0]
except:
    dataset = "usps0"

try:
    X_train, y_train, X_test, y_test = load_dataset(dataset)
except KeyError:
    raise ValueError("Wrong dataset name!")

n_samples = X_train.shape[0]

if X_test is None:
    X_test, y_test = X_train, y_train

Cs = np.linspace(0.1, 1, 10)
#Cs = np.linspace(0.1, 5, 10)

res_p = [fit_primal_svc(X_train, y_train, C=C, kernel=opts.kernel,
                        gamma=opts.gamma, degree=opts.degree) for C in Cs]
clfs_p, train_times_p = zip(*res_p)
res_p = [predict(clf, X_test, y_test) for clf in clfs_p]
accuracies_p, test_times_p = zip(*res_p)

res_d = [fit_dual_svc(X_train, y_train, C=C, kernel=opts.kernel,
                      gamma=opts.gamma, degree=opts.degree) for C in Cs]
clfs_d, train_times_d = zip(*res_d)
res_d = [predict(clf, X_test, y_test) for clf in clfs_d]
accuracies_d, test_times_d = zip(*res_d)

opt = {"linewidth":1, "markersize":15, "markerfacecolor":'None'}

if opts.bw:
    style = ("k-", "k--")
else:
    style = ("b-", "g--")

prop = fm.FontProperties(size=18)

pl.figure()
set_axes_size(pl)
pl.plot(Cs, [np.mean(1.0 * clf.n_support_vectors() / n_samples) for clf in clfs_p],
        style[0], label="L2L L1R Primal", **opt)
pl.plot(Cs, [np.mean(1.0 * clf.n_support_vectors() / n_samples) for clf in clfs_d],
        style[1], label="L1L L2R Dual", **opt)
pl.xlabel('C', size=15)
pl.ylabel('Percentage of components / support vectors', size=15)
pl.legend(loc='lower right', prop=prop)
if not opts.notitle:
    pl.title('Relation between C and sparsity')

pl.figure()
set_axes_size(pl)
pl.plot(Cs, train_times_p, style[0], label="L2L L1R Primal", **opt)
pl.plot(Cs, train_times_d, style[1], label="L1L L2R Dual", **opt)
pl.xlabel('C', size=15)
pl.ylabel('Training time in seconds', size=15)
if not opts.notitle:
    pl.title('Relation between C and training time')

pl.figure()
set_axes_size(pl)
pl.plot(Cs, accuracies_p, style[0], label="L2L L1R Primal", **opt)
pl.plot(Cs, accuracies_d, style[1], label="L1L L2R Dual", **opt)
pl.xlabel('C', size=15)
pl.ylabel('Test accuracy', size=15)
if not opts.notitle:
    pl.title('Relation between C and accuracy')

pl.show()
