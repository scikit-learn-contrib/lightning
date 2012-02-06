# Author: Mathieu Blondel
# License: BSD
"""
=======================================
Kernel Matching Pursuit semi-supervised
=======================================

"""
print __doc__

import sys

import numpy as np
import pylab as pl

from sklearn.utils import check_random_state
from sklearn.linear_model import Ridge
from sklearn.metrics import f1_score, precision_score, recall_score

from lightning.datasets import load_dataset
from lightning.kmp import KMPClassifier, KMPRegressor
from lightning.kmp import select_components, create_components

from sklearn.externals.joblib import Memory
from lightning.datasets import get_data_home

from common import parse_kmp, plot, split_data

memory = Memory(cachedir=get_data_home(), verbose=0, compress=6)

def split_unlabeled_data(X_train, y_train, n_components, perc_label, random_state):
    random_state = check_random_state(random_state)
    n_samples = X_train.shape[0]

    if 0 < n_components and n_components <= 1.0:
        n_components = int(n_components * n_samples)

    indices = np.arange(n_samples)
    random_state.shuffle(indices)
    indices = indices[:n_components]

    n_label = int(n_components * perc_label)

    X_all = X_train[indices]
    X_l = X_train[:n_label]
    y_l = y_train[:n_label]

    return X_all, X_l, y_l


@memory.cache
def fit_kmp(X_train, y_train, X_test, y_test, components, opts, random_state):
    klass = KMPRegressor if opts.regression else KMPClassifier
    clf = klass(n_nonzero_coefs=opts.n_nonzero_coefs,
                init_components=components,
                n_refit=opts.n_refit,
                estimator=Ridge(alpha=opts.alpha),
                metric=opts.metric,
                gamma=opts.gamma,
                degree=opts.degree,
                coef0=opts.coef0,
                scale=opts.scale,
                scale_y=opts.scale_y,
                check_duplicates=opts.check_duplicates,
                n_validate=opts.n_validate,
                epsilon=opts.epsilon,
                X_val=X_test, y_val=y_test,
                verbose=1,
                random_state=random_state,
                n_jobs=-1)
    clf.fit(X_train, y_train)
    return clf

X_train, y_train, X_test, y_test, opts, args = parse_kmp(n_components=1.0,
                                                         check_duplicates=True)
X_tr, y_tr, X_te, y_te = X_train, y_train, X_test, y_test

if opts.n_nonzero_coefs < 1:
    raise ValueError("n_nonzero_coefs must be a positive integer")

cv = list(split_data(X_train, y_train,
                     X_test, y_test,
                     opts.n_folds,
                     not opts.regression))

amounts = np.linspace(0.1, 1.0, 10)
#amounts = (0.25, 0.5, 0.75, 1.0)
#amounts = np.linspace(0.1, 0.5, 5)
acc_sup = np.zeros((len(amounts), len(cv)), dtype=np.float64)
acc_semi = np.zeros((len(amounts), len(cv)), dtype=np.float64)

j = 0
for X_tr, y_tr, X_te, y_te in cv:

    for i, perc_label in enumerate(amounts):
        print "Percentage of labeled data:", perc_label
        X_all, X_l, y_l = split_unlabeled_data(X_tr,
                                               y_tr,
                                               n_components=opts.n_components,
                                               perc_label=perc_label,
                                               random_state=0)

        clf = fit_kmp(X_l, y_l, X_te, y_te, X_l, opts, 0)
        #acc_sup[i, j] = clf.validation_scores_[-1]
        acc_sup[i, j] = clf.best_score_

        if perc_label == 1.0:
            acc_semi[i, j] = acc_sup[i, j]
        else:
            clf = fit_kmp(X_l, y_l, X_te, y_te, X_all, opts, 0)
            #acc_semi[i, j] = clf.validation_scores_[-1]
            acc_semi[i, j] = clf.best_score_

    j += 1

error_bar = len(args) == 2

# 2-d plot
pl.figure()

plot(pl, amounts,
     acc_sup.mean(axis=1), acc_sup.std(axis=1),
     "Supervised", error_bar)
plot(pl, amounts,
     acc_semi.mean(axis=1), acc_semi.std(axis=1),
     "Semi-supervised", error_bar)

pl.xlabel('Percentage of labeled data')

if opts.regression:
    pl.ylabel('MSE')
    pl.legend(loc='upper right')
else:
    pl.ylabel('Accuracy')
    pl.legend(loc='lower right')

pl.show()

