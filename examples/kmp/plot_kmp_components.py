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

from lightning.datasets import load_dataset
from lightning.kmp import KMPClassifier, KMPRegressor
from lightning.kmp import select_components, create_components

from sklearn.externals.joblib import Memory
from lightning.datasets import get_data_home

memory = Memory(cachedir=get_data_home(), verbose=0, compress=6)

from common import parse_kmp, plot, split_data

options = ("n_nonzero_coefs", "metric",
           "gamma", "degree", "coef0", "scale", "scale_y",
           "check_duplicates", "n_validate", "epsilon")

def options_to_dict(opts):
    d = {}
    for opt in options:
        d[opt] = getattr(opts, opt)
    return d

@memory.cache
def create_kmeans_comp(X_train, y_train, class_distrib, n_components,
                       random_state):
    return create_components(X_train, y_train, n_components=n_components,
                             class_distrib=class_distrib,
                             random_state=random_state,
                             verbose=1)


@memory.cache
def fit_kmp(X_train, y_train, X_test, y_test, components, opt_dict, regression,
            random_state):
    klass = KMPRegressor if regression else KMPClassifier
    clf = klass(init_components=components,
                estimator=Ridge(alpha=0.1),
                X_val=X_test, y_val=y_test,
                verbose=1,
                random_state=random_state,
                n_jobs=-1,
                **opt_dict)
    clf.fit(X_train, y_train)
    return clf

X_train, y_train, X_test, y_test, opts, args = parse_kmp(check_duplicates=True)
opt_dict = options_to_dict(opts)

class_distrib = "random" if opts.regression else "balanced"

clf_s = []
clf_kg = []
clf_kb = []
clf_ks = []

j = 0
for X_tr, y_tr, X_te, y_te in split_data(X_train, y_train,
                                         X_test, y_test,
                                         opts.n_folds,
                                         opts.cvtype,
                                         opts.force_cv):
    print "Fold", j

    # selected from datasets
    print "Selected components"
    components = select_components(X_tr, y_tr, opts.n_components,
                                   class_distrib=class_distrib, random_state=j)
    clf_s.append(fit_kmp(X_tr, y_tr, X_te, y_te, components, opt_dict,
                         opts.regression, random_state=j))

    # k-means global
    print "Global k-means"
    components = create_kmeans_comp(X_tr, y_tr,
                                    n_components=opts.n_components,
                                    class_distrib="global",
                                    random_state=j)
    clf_kg.append(fit_kmp(X_tr, y_tr, X_te, y_te, components, opt_dict,
                          opts.regression, random_state=j))

    if not opts.regression:
        # k-means balanced
        print "Balanced k-means"
        components = create_kmeans_comp(X_tr, y_tr,
                                        n_components=opts.n_components,
                                        class_distrib="balanced",
                                        random_state=j)
        clf_kb.append(fit_kmp(X_tr, y_tr, X_te, y_te, components, opt_dict,
                              opts.regression, random_state=j))

        # k-means stratified
        print "Stratified k-means"
        components = create_kmeans_comp(X_tr, y_tr,
                                        n_components=opts.n_components,
                                        class_distrib="stratified",
                                        random_state=j)
        clf_ks.append(fit_kmp(X_tr, y_tr, X_te, y_te, components, opt_dict,
                              opts.regression, random_state=j))

    j += 1

ss = np.vstack([clf.validation_scores_ for clf in clf_s])
kgs = np.vstack([clf.validation_scores_ for clf in clf_kg])

if not opts.regression:
    kbs = np.vstack([clf.validation_scores_ for clf in clf_kb])
    kss = np.vstack([clf.validation_scores_ for clf in clf_ks])

pl.figure()

plot(pl, clf_s[0].iterations_,
     ss.mean(axis=0), ss.std(axis=0),
     "Selected", opts.bars)
plot(pl, clf_kg[0].iterations_,
     kgs.mean(axis=0), kgs.std(axis=0),
     "K-means global", opts.bars)

if not opts.regression:
    plot(pl, clf_kb[0].iterations_,
         kbs.mean(axis=0), kbs.std(axis=0),
         "K-means balanced", opts.bars)
    plot(pl, clf_ks[0].iterations_,
         kss.mean(axis=0), kss.std(axis=0),
         "K-means stratified", opts.bars)

pl.xlabel('Iteration')

if opts.regression:
    pl.ylabel('MSE')
    pl.legend(loc='upper right')
else:
    pl.ylabel('Accuracy')
    pl.legend(loc='lower right')

if len(opts.savefig) > 0:
    pl.savefig(opts.savefig)
else:
    pl.show()
