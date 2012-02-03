# Author: Mathieu Blondel
# License: BSD
"""
=================================
Kernel Matching Pursuit accuracy
=================================

"""
print __doc__

import numpy as np
import pylab as pl

from sklearn.linear_model import Ridge

from lightning.datasets import load_dataset, split_data
from lightning.kmp import KMPClassifier, KMPRegressor

from sklearn.externals.joblib import Memory
from lightning.datasets import get_data_home

from common import parse_kmp

memory = Memory(cachedir=get_data_home(), verbose=0, compress=6)


@memory.cache
def fit_kmp(X_train, y_train, X_test, y_test, opts, random_state):
    klass = KMPRegressor if opts.regression else KMPClassifier
    clf = klass(n_nonzero_coefs=opts.n_nonzero_coefs,
                n_components=opts.n_components,
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
                verbose=2,
                random_state=random_state,
                n_jobs=-1)
    clf.fit(X_train, y_train)
    return clf

X_train, y_train, X_test, y_test, opts = parse_kmp()

validation_scores = 0.0
training_scores = 0.0

for i in range(opts.n_times):
    if X_test is None:
        X_train, y_train, X_test, y_test = split_data(X_train, y_train,
                                                      proportion_train=0.75,
                                                      random_state=i)

    clf = fit_kmp(X_train, y_train, X_test, y_test, opts, random_state=i)
    validation_scores += clf.validation_scores_
    training_scores += clf.training_scores_

validation_scores /= opts.n_times
training_scores /= opts.n_times

pl.figure()
pl.plot(clf.iterations_, validation_scores, label="Test set")
pl.plot(clf.iterations_, training_scores, label="Training set")
pl.xlabel('Iteration')

if opts.regression:
    pl.ylabel('MSE')
    pl.legend(loc='upper right')
else:
    pl.ylabel('Accuracy')
    pl.legend(loc='lower right')

pl.show()
