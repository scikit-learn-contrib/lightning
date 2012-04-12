# Author: Mathieu Blondel
# License: BSD

import sys
import time
from optparse import OptionParser

import numpy as np
import pylab as pl
import matplotlib.font_manager as fm

from sklearn.metrics.pairwise import pairwise_kernels

from lightning.dual_cd import DualSVC
from lightning.primal_cd import PrimalSVC
from lightning.datasets import load_dataset

def set_axes_size(pl):
    ax = pl.axes()
    for xlabel in ax.get_xticklabels():
        xlabel.set_fontsize(13)

    for ylabel in ax.get_yticklabels():
        ylabel.set_fontsize(13)


class Callback(object):

    def __init__(self, K_test, y_test):
        self.n_svs = []
        self.times = []
        self.accuracies = []
        self.K_test = K_test
        self.y_test = y_test

    def __call__(self, clf):
        self.n_svs.append(clf.n_support_vectors())
        if isinstance(clf, PrimalSVC):
            pred = np.dot(self.K_test, clf.coef_.T)
        else:
            pred = np.dot(self.K_test, clf.dual_coef_.T)
        pred = clf.label_binarizer_.inverse_transform(pred)
        acc = np.mean(pred == y_test)
        self.accuracies.append(acc)
        self.times.append(time.time())

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

K_test = pairwise_kernels(X_test, X_train, metric="rbf", gamma=0.1)

callback_primal = Callback(K_test, y_test)
clf = PrimalSVC(kernel="rbf", gamma=0.1, penalty="l1",
                callback=callback_primal, C=1.0, max_iter=10, tol=1e-4)
clf.fit(X_train, y_train)

callback_dual = Callback(K_test, y_test)
clf = DualSVC(kernel="rbf", gamma=0.1, loss="l1",
              callback=callback_dual, C=1.0, max_iter=10, tol=1e-4)
clf.fit(X_train, y_train)

opt = {"linewidth":1, "markersize":15, "markerfacecolor":'None'}

if opts.bw:
    style = ("k-", "k--")
else:
    style = ("b-", "g--")

prop = fm.FontProperties(size=18)

pl.figure()
set_axes_size(pl)


pl.plot(np.array(callback_primal.times) - callback_primal.times[0],
        np.array(callback_primal.n_svs) * 1.0 / n_samples,
        style[0], label="L2L L1R Primal", **opt)
pl.plot(np.array(callback_dual.times) - callback_dual.times[0],
        np.array(callback_dual.n_svs) * 1.0 / n_samples,
        style[1], label="L1L L2R Dual", **opt)
pl.ylim([0.0, 0.8])
pl.xlabel('CPU seconds', size=15)
pl.ylabel('Percentage of components / support vectors', size=15)
pl.legend(loc='lower right', prop=prop)
if not opts.notitle:
    pl.title('Sparsity over time')

pl.figure()
set_axes_size(pl)


pl.plot(np.array(callback_primal.times) - callback_primal.times[0],
        np.array(callback_primal.accuracies),
        style[0], label="L2L L1R Primal", **opt)
pl.plot(np.array(callback_dual.times) - callback_dual.times[0],
        np.array(callback_dual.accuracies),
        style[1], label="L1L L2R Dual", **opt)
pl.ylim([0.97, 1.0])
pl.xlabel('CPU seconds', size=15)
pl.ylabel('Test accuracy', size=15)
if not opts.notitle:
    pl.title('Accuracy over time')

pl.show()
