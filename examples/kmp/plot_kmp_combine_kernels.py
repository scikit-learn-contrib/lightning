# Author: Mathieu Blondel
# License: BSD
"""
========================================
Kernel Matching Pursuit combine kernels
========================================

"""
print __doc__

import numpy as np
import pylab as pl

from sklearn.linear_model import Ridge
from sklearn.metrics import pairwise_kernels

from lightning.datasets import load_dataset
from lightning.kmp import KMPClassifier, KMPRegressor, select_components

from sklearn.externals.joblib import Memory
from lightning.datasets import get_data_home

from common import parse_kmp

memory = Memory(cachedir=get_data_home(), verbose=0, compress=6)

@memory.cache
def fit_kmp(K_train, y_train, K_test, y_test, opts, random_state):
    klass = KMPRegressor if opts.regression else KMPClassifier
    clf = klass(n_nonzero_coefs=opts.n_nonzero_coefs,
                n_refit=opts.n_refit,
                estimator=Ridge(alpha=opts.alpha),
                X_val=K_test, y_val=y_test,
                metric="precomputed",
                scale=True,
                scale_y=opts.scale_y,
                check_duplicates=opts.check_duplicates,
                n_validate=opts.n_validate,
                epsilon=opts.epsilon,
                verbose=1,
                random_state=random_state,
                n_jobs=-1)
    clf.fit(K_train, y_train)
    return clf

dataset, opts, random_state = parse_kmp()

try:
    X_train, y_train, X_test, y_test = load_dataset(dataset,
                                                    proportion_train=0.75,
                                                    random_state=random_state)
except KeyError:
    raise ValueError("Wrong dataset name!")

print "X_train", X_train.shape
print "X_test", X_test.shape

class_distrib = "random" if opts.regression else "balanced"

components = select_components(X_train, y_train,
                               n_components=opts.n_components,
                               class_distrib=class_distrib)

print "Computing linear kernels..."
linear_train = pairwise_kernels(X_train, components, metric="linear")
linear_test = pairwise_kernels(X_test, components, metric="linear")

print "Computing rbf kernels..."
rbf_train = pairwise_kernels(X_train, components, metric="rbf",
                             gamma=opts.gamma)
rbf_test = pairwise_kernels(X_test, components, metric="rbf",
                            gamma=opts.gamma)

print "Computing polynomial kernels..."
poly_train = pairwise_kernels(X_train, components, metric="poly",
                              degree=opts.degree)
poly_test = pairwise_kernels(X_test, components, metric="poly",
                              degree=opts.degree)

n_components = components.shape[0]

print "Combining kernels..."
n = n_components / 3
combined3_train = np.hstack((linear_train[:, :n],
                             rbf_train[:, :n],
                             poly_train[:, :n]))
combined3_test = np.hstack((linear_test[:, :n],
                            rbf_test[:, :n],
                            poly_test[:, :n]))

clf_l = fit_kmp(linear_train, y_train,
                linear_test, y_test,
                opts, 0)

clf_r = fit_kmp(rbf_train, y_train,
                rbf_test, y_test,
                opts, 0)

clf_p = fit_kmp(poly_train, y_train,
                poly_test, y_test,
                opts, 0)

clf_c3 = fit_kmp(combined3_train, y_train,
                 combined3_test, y_test,
                 opts, 0)

pl.figure()
pl.plot(clf_l.iterations_, clf_l.validation_scores_, label="Linear kernel")
pl.plot(clf_r.iterations_, clf_r.validation_scores_, label="RBF kernel")
pl.plot(clf_p.iterations_, clf_p.validation_scores_, label="Poly kernel")
pl.plot(clf_c3.iterations_, clf_c3.validation_scores_, label="Combined 3 kernels")
pl.xlabel('Iteration')

if opts.regression:
    pl.ylabel('MSE')
    pl.legend(loc='upper right')
else:
    pl.ylabel('Accuracy')
    pl.legend(loc='lower right')

pl.show()
