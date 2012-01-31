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

from lightning.datasets import load_dataset
from lightning.kmp import KMPClassifier

from sklearn.externals.joblib import Memory
from lightning.datasets import get_data_home

from parser import parse_kmp

memory = Memory(cachedir=get_data_home(), verbose=0, compress=6)


@memory.cache
def fit_kmp(X_train, y_train, X_test, y_test, opts, random_state):
    clf = KMPClassifier(n_nonzero_coefs=opts.n_nonzero_coefs,
                        n_components=opts.n_components,
                        n_refit=opts.n_refit,
                        estimator=Ridge(alpha=opts.alpha),
                        X_val=X_test, y_val=y_test,
                        metric=opts.metric,
                        gamma=opts.gamma,
                        degree=opts.degree,
                        coef0=opts.coef0,
                        scale=opts.scale,
                        check_duplicates=opts.check_duplicates,
                        n_validate=opts.n_validate,
                        epsilon=opts.epsilon,
                        verbose=2,
                        random_state=random_state,
                        n_jobs=-1)
    clf.fit(X_train, y_train)
    return clf

dataset, opts, random_state = parse_kmp(epsilon=0.001)

try:
    X_train, y_train, X_test, y_test = load_dataset(dataset,
                                                    proportion_train=0.75,
                                                    random_state=random_state)
except KeyError:
    raise ValueError("Wrong dataset name!")

clf = fit_kmp(X_train, y_train, X_test, y_test, opts, random_state)

pl.figure()
pl.plot(clf.iterations_, clf.validation_scores_, label="Test set")
pl.plot(clf.iterations_, clf.training_scores_, label="Training set")
pl.xlabel('Iteration')
pl.ylabel('Accuracy')
pl.title('Accuracy plot')
pl.legend()

pl.show()
