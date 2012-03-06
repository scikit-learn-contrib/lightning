# Author: Mathieu Blondel
# License: BSD

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_random_state, safe_mask
from sklearn.metrics.pairwise import pairwise_kernels

from .kernel_fast import get_kernel
from .lasvm_fast import _lasvm

class LaSVM(BaseEstimator, ClassifierMixin):

    def __init__(self, C=1.0, max_iter=10, tol=1e-3,
                 kernel="linear", gamma=0.1, coef0=1, degree=4,
                 warm_start=False, random_state=None, verbose=0, n_jobs=1):
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.kernel = kernel
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
        self.warm_start = warm_start
        self.random_state = random_state
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.support_vectors_ = None
        self.dual_coef_ = None

    def _kernel_params(self):
        return {"gamma" : self.gamma,
                "degree" : self.degree,
                "coef0" : self.coef0}

    def _get_kernel(self):
        return get_kernel(self.kernel, **self._kernel_params())

    def fit(self, X, y):
        n_samples = X.shape[0]
        rs = check_random_state(self.random_state)

        self.label_binarizer_ = LabelBinarizer(neg_label=-1, pos_label=1)
        Y = self.label_binarizer_.fit_transform(y)
        n_vectors = Y.shape[1]

        warm_start = False
        if self.warm_start and self.dual_coef_ is not None:
            warm_start = True

        if not self.warm_start or self.dual_coef_ is None:
            self.dual_coef_ = np.zeros((n_vectors, n_samples), dtype=np.float64)

        self.intercept_ = np.zeros((n_vectors,), dtype=np.float64)
        kernel = self._get_kernel()

        for i in xrange(n_vectors):
            b = _lasvm(self.dual_coef_[i],
                       X, Y[:, i], kernel,
                       self.C, self.max_iter, rs, self.tol,
                       verbose=self.verbose,
                       warm_start=warm_start)
            self.intercept_[i] = b

        sv = np.sum(self.dual_coef_ != 0, axis=0, dtype=bool)

        if self.kernel != "precomputed":
            if not self.warm_start:
                self.dual_coef_ = self.dual_coef_[:, sv]
                mask = safe_mask(X, sv)
                self.support_vectors_ = X[mask]
            else:
                # Cannot trim the non-zero weights if warm start is used...
                self.support_vectors_ = X

        if self.verbose >= 1:
            print "Number of support vectors:", np.sum(sv)

        return self

    def decision_function(self, X):
        K = pairwise_kernels(X, self.support_vectors_, metric=self.kernel,
                             filter_params=True, n_jobs=self.n_jobs,
                             **self._kernel_params())
        return np.dot(K, self.dual_coef_.T) + self.intercept_

    def predict(self, X):
        pred = self.decision_function(X)
        return self.label_binarizer_.inverse_transform(pred, threshold=0)
