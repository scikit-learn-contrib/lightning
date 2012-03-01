# Author: Mathieu Blondel
# License: BSD

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_random_state, safe_mask
from sklearn.metrics.pairwise import pairwise_kernels

from dual_cd_fast import _dual_cd

class DualLinearSVC(BaseEstimator, ClassifierMixin):

    def __init__(self, C=1.0, loss="l1", max_iter=1000, tol=1e-4,
                 random_state=None, verbose=0, n_jobs=1):
        self.C = C
        self.loss = loss
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose
        self.n_jobs = n_jobs

    def fit(self, X, y):
        n_features = X.shape[1]
        rs = check_random_state(self.random_state)
        self.label_binarizer_ = LabelBinarizer(neg_label=-1, pos_label=1)
        Y = self.label_binarizer_.fit_transform(y)
        n_vectors = Y.shape[1]
        self.coef_ = np.zeros((n_vectors, n_features), dtype=np.float64)

        for i in xrange(n_vectors):
            _dual_cd(self.coef_[i], X, Y[:, i],
                     self.C, self.loss, self.max_iter, rs, self.tol,
                     precomputed_kernel=False, verbose=self.verbose)

        return self

    def decision_function(self, X):
        return np.dot(X, self.coef_.T)

    def predict(self, X):
        pred = self.decision_function(X)
        return self.label_binarizer_.inverse_transform(pred, threshold=0)


class DualSVC(BaseEstimator, ClassifierMixin):

    def __init__(self, C=1.0, loss="l1", max_iter=1000, tol=1e-4,
                 kernel="linear", gamma=0.1, coef0=1, degree=4,
                 random_state=None, verbose=0, n_jobs=1):
        self.C = C
        self.loss = loss
        self.max_iter = max_iter
        self.tol = tol
        self.kernel = kernel
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
        self.random_state = random_state
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.support_vectors_ = None

    def _kernel_params(self):
        return {"gamma" : self.gamma,
                "degree" : self.degree,
                "coef0" : self.coef0}

    def fit(self, X, y):
        n_samples = X.shape[0]
        rs = check_random_state(self.random_state)

        self.label_binarizer_ = LabelBinarizer(neg_label=-1, pos_label=1)
        Y = self.label_binarizer_.fit_transform(y)
        n_vectors = Y.shape[1]

        K = pairwise_kernels(X, X, metric=self.kernel,
                             filter_params=True, n_jobs=self.n_jobs,
                             **self._kernel_params())
        self.dual_coef_ = np.zeros((n_vectors, n_samples), dtype=np.float64)

        for i in xrange(n_vectors):
            _dual_cd(self.dual_coef_[i], K, Y[:, i],
                     self.C, self.loss, self.max_iter, rs, self.tol,
                     precomputed_kernel=True, verbose=self.verbose)

        self.dual_coef_ *= Y.T

        if self.kernel != "precomputed":
            sv = np.sum(self.dual_coef_ != 0, axis=0, dtype=bool)
            self.dual_coef_ = self.dual_coef_[:, sv]
            mask = safe_mask(X, sv)
            self.support_vectors_ = X[mask]

        if self.verbose >= 1:
            print "Number of support vectors:", np.sum(sv)

        return self

    def decision_function(self, X):
        K = pairwise_kernels(X, self.support_vectors_, metric=self.kernel,
                             filter_params=True, n_jobs=self.n_jobs,
                             **self._kernel_params())
        return np.dot(K, self.dual_coef_.T)

    def predict(self, X):
        pred = self.decision_function(X)
        return self.label_binarizer_.inverse_transform(pred, threshold=0)
