# Author: Mathieu Blondel
# License: BSD
"""
=====================================
Kernel Matching Pursuit combine views
=====================================

"""
print __doc__

import numpy as np
import pylab as pl

from sklearn.linear_model import Ridge
from sklearn.decomposition import RandomizedPCA
from sklearn.metrics import pairwise_kernels

from lightning.datasets import load_dataset
from lightning.kmp import KMPClassifier, select_components

from sklearn.externals.joblib import Memory
from lightning.datasets import get_data_home

from common import parse_kmp

memory = Memory(cachedir=get_data_home(), verbose=0, compress=6)

@memory.cache
def fit_kmp(K_train, y_train, K_test, y_test, opts, random_state):
    clf = KMPClassifier(n_nonzero_coefs=opts.n_nonzero_coefs,
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

# PCA view
print "Computing PCA..."
pca = RandomizedPCA(n_components=300)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

components_pca = select_components(X_train_pca, y_train,
                                   n_components=opts.n_components,
                                   class_distrib="balanced")

print "Computing kernels (PCA view)..."
K_pca_train = pairwise_kernels(X_train_pca, components_pca, metric="rbf",
                               gamma=0.1)
K_pca_test = pairwise_kernels(X_test_pca, components_pca, metric="rbf",
                              gamma=0.1)

# Regular view
components = select_components(X_train, y_train,
                               n_components=opts.n_components,
                               class_distrib="balanced")

print "Computing kernels (regular view)..."
K_train = pairwise_kernels(X_train, components, metric="rbf", gamma=0.1)
K_test = pairwise_kernels(X_test, components, metric="rbf", gamma=0.1)

# Combined views
n_components = components.shape[0]
n = n_components / 2
K_combined_train = np.hstack((K_pca_train[:, :n],
                              K_train[:, :n]))
K_combined_test = np.hstack((K_pca_test[:, :n],
                             K_test[:, :n]))

clf_p = fit_kmp(K_pca_train, y_train,
                K_pca_test, y_test,
                opts, 0)

clf_r = fit_kmp(K_train, y_train,
                K_test, y_test,
                opts, 0)

clf_c = fit_kmp(K_combined_train, y_train,
                K_combined_test, y_test,
                opts, 0)

pl.figure()
pl.plot(clf_p.iterations_, clf_p.validation_scores_, label="PCA view")
pl.plot(clf_r.iterations_, clf_r.validation_scores_, label="Regular view")
pl.plot(clf_c.iterations_, clf_c.validation_scores_, label="Combined views")
pl.xlabel('Iteration')

pl.ylabel('Accuracy')
pl.legend(loc='lower right')

pl.show()
