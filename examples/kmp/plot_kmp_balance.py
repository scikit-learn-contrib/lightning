# Author: Mathieu Blondel
# License: BSD
"""
=================================
Kernel Matching Pursuit balance
=================================

"""
print __doc__

import numpy as np
import pylab as pl

from sklearn.linear_model import Ridge
from sklearn.metrics import f1_score, precision_score, recall_score

from lightning.datasets import load_dataset
from lightning.kmp import KMPClassifier, select_components

from sklearn.externals.joblib import Memory
from lightning.datasets import get_data_home

from common import parse_kmp, plot, split_data

memory = Memory(cachedir=get_data_home(), verbose=0, compress=6)


@memory.cache
def fit_kmp(X_train, y_train, X_test, y_test, class_distrib, opts, random_state):
    components = select_components(X_train, y_train,
                                   n_components=opts.n_components,
                                   class_distrib=class_distrib,
                                   random_state=random_state)

    clf = KMPClassifier(n_nonzero_coefs=opts.n_nonzero_coefs,
                        init_components=components,
                        n_refit=opts.n_refit,
                        estimator=Ridge(alpha=opts.alpha),
                        X_val=X_test, y_val=y_test,
                        metric=opts.metric,
                        gamma=opts.gamma,
                        degree=opts.degree,
                        coef0=opts.coef0,
                        scale=opts.scale,
                        n_validate=opts.n_validate,
                        epsilon=opts.epsilon,
                        #score_func=f1_score,
                        verbose=1,
                        random_state=random_state,
                        n_jobs=-1)
    clf.fit(X_train, y_train)
    return clf


X_train, y_train, X_test, y_test, opts, args = parse_kmp()
X_tr, y_tr, X_te, y_te = X_train, y_train, X_test, y_test

clf_r = []
clf_b = []
clf_s = []

for X_tr, y_tr, X_te, y_te in split_data(X_train, y_train,
                                         X_test, y_test,
                                         opts.n_folds,
                                         not opts.regression):

    clf_r.append(fit_kmp(X_tr, y_tr, X_te, y_te, "random", opts,
                         random_state=0))
    clf_b.append(fit_kmp(X_tr, y_tr, X_te, y_te, "balanced", opts,
                    random_state=0))
    clf_s.append(fit_kmp(X_tr, y_tr, X_te, y_te, "stratified", opts,
                    random_state=0))

rs = np.vstack([clf.validation_scores_ for clf in clf_r])
bs = np.vstack([clf.validation_scores_ for clf in clf_b])
ss = np.vstack([clf.validation_scores_ for clf in clf_s])

pl.figure()

error_bar = len(args) == 2
plot(pl, clf_r[0].iterations_,
     rs.mean(axis=0), rs.std(axis=0),
     "Random", error_bar)
plot(pl, clf_b[0].iterations_,
     bs.mean(axis=0), bs.std(axis=0),
     "Balanced", error_bar)
plot(pl, clf_s[0].iterations_,
     ss.mean(axis=0), ss.std(axis=0),
     "Stratified", error_bar)

pl.xlabel('Iteration')
pl.ylabel('Accuracy')
pl.legend(loc='lower right')

pl.show()
