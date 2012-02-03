# Author: Mathieu Blondel
# License: BSD
"""
==================================
Kernel Matching Pursuit components
==================================

"""
print __doc__

import sys

import numpy as np
import pylab as pl

from sklearn.linear_model import Ridge
from sklearn.metrics import f1_score, precision_score, recall_score

from lightning.datasets import load_dataset, split_data
from lightning.kmp import KMPClassifier, KMPRegressor
from lightning.kmp import select_components, create_components

from sklearn.externals.joblib import Memory
from lightning.datasets import get_data_home

memory = Memory(cachedir=get_data_home(), verbose=0, compress=6)

from common import parse_kmp, plot

@memory.cache
def create_kmeans_comp(X_train, y_train, class_distrib, n_components,
                       random_state):
    return create_components(X_train, y_train, n_components=n_components,
                             class_distrib=class_distrib,
                             random_state=random_state,
                             verbose=1)


@memory.cache
def fit_kmp(X_train, y_train, X_test, y_test, components, random_state):
    klass = KMPRegressor if opts.regression else KMPClassifier
    clf = klass(n_nonzero_coefs=opts.n_nonzero_coefs,
                init_components=components,
                n_refit=opts.n_refit,
                estimator=Ridge(alpha=opts.alpha),
                X_val=X_test, y_val=y_test,
                metric=opts.metric,
                gamma=opts.gamma,
                degree=opts.degree,
                coef0=opts.coef0,
                scale=opts.scale,
                scale_y=opts.scale_y,
                check_duplicates=opts.check_duplicates,
                n_validate=opts.n_validate,
                epsilon=opts.epsilon,
                verbose=1,
                random_state=random_state,
                n_jobs=-1)
    clf.fit(X_train, y_train)
    return clf

X_train, y_train, X_test, y_test, opts, args = parse_kmp()
X_tr, y_tr, X_te, y_te = X_train, y_train, X_test, y_test

class_distrib = "random" if opts.regression else "balanced"

clf_s = []
clf_kg = []
clf_kb = []
clf_ks = []

for i in range(opts.n_times):
    if X_test is None:
        X_tr, y_tr, X_te, y_te = split_data(X_train, y_train,
                                            proportion_train=0.75,
                                            random_state=i)

    # selected from datasets
    components = select_components(X_tr, y_tr, opts.n_components,
                                   class_distrib=class_distrib, random_state=i)
    clf_s.append(fit_kmp(X_tr, y_tr, X_te, y_te, components,
                         random_state=i))

    # k-means global
    components = create_kmeans_comp(X_tr, y_tr,
                                    n_components=opts.n_components,
                                    class_distrib="global",
                                    random_state=i)
    clf_kg.append(fit_kmp(X_tr, y_tr, X_te, y_te, components,
                          random_state=i))

    if not opts.regression:
        # k-means balanced
        components = create_kmeans_comp(X_tr, y_tr,
                                        n_components=opts.n_components,
                                        class_distrib="balanced",
                                        random_state=i)
        clf_kb.append(fit_kmp(X_tr, y_tr, X_te, y_te, components,
                              random_state=i))

        # k-means stratified
        components = create_kmeans_comp(X_tr, y_tr,
                                        n_components=opts.n_components,
                                        class_distrib="stratified",
                                        random_state=i)
        clf_ks.append(fit_kmp(X_tr, y_tr, X_te, y_te, components,
                              random_state=i))

ss = np.vstack([clf.validation_scores_ for clf in clf_s])
kgs = np.vstack([clf.validation_scores_ for clf in clf_kg])

if not opts.regression:
    kbs = np.vstack([clf.validation_scores_ for clf in clf_kb])
    kss = np.vstack([clf.validation_scores_ for clf in clf_ks])

error_bar = len(args) == 2

pl.figure()

plot(pl, clf_s[0].iterations_,
     ss.mean(axis=0), ss.std(axis=0),
     "Selected", error_bar)
plot(pl, clf_kg[0].iterations_,
     kgs.mean(axis=0), kgs.std(axis=0),
     "K-means global", error_bar)

if not opts.regression:
    plot(pl, clf_kb[0].iterations_,
         kbs.mean(axis=0), kbs.std(axis=0),
         "K-means balanced", error_bar)
    plot(pl, clf_ks[0].iterations_,
         kss.mean(axis=0), kss.std(axis=0),
         "K-means stratified", error_bar)

pl.xlabel('Iteration')

if opts.regression:
    pl.ylabel('MSE')
    pl.legend(loc='upper right')
else:
    pl.ylabel('Accuracy')
    pl.legend(loc='lower right')

pl.show()
