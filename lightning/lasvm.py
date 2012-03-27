# Author: Mathieu Blondel
# License: BSD

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_random_state, safe_mask
from sklearn.metrics.pairwise import pairwise_kernels

from .kernel_fast import get_kernel, KernelCache
from .lasvm_fast import _lasvm
from .predict_fast import predict_alpha, decision_function_alpha

class LaSVM(BaseEstimator, ClassifierMixin):

    def __init__(self, C=1.0, max_iter=10,
                 kernel="linear", gamma=0.1, coef0=1, degree=4,
                 selection="permute", search_size=60,
                 termination="n_iter", sv_upper_bound=1000,
                 tau=1e-3, finish_step=True,
                 warm_start=False, cache_mb=500,
                 random_state=None, callback=None, verbose=0, n_jobs=1):
        self.C = C
        self.max_iter = max_iter
        self.kernel = kernel
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
        self.selection = selection
        self.search_size = search_size
        self.termination = termination
        self.sv_upper_bound = sv_upper_bound
        self.tau = tau
        self.finish_step = finish_step
        self.warm_start = warm_start
        self.random_state = random_state
        self.cache_mb = cache_mb
        self.callback = callback
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
        self.classes_ = self.label_binarizer_.classes_.astype(np.int32)
        n_vectors = Y.shape[1]

        warm_start = False
        if self.warm_start and self.dual_coef_ is not None:
            warm_start = True

        if not self.warm_start or self.dual_coef_ is None:
            self.dual_coef_ = np.zeros((n_vectors, n_samples), dtype=np.float64)

        self.intercept_ = np.zeros((n_vectors,), dtype=np.float64)
        kernel = self._get_kernel()
        kcache = KernelCache(kernel, n_samples, self.cache_mb * 1024 * 1024,
                             self.verbose)
        self.support_vectors_ = X

        for i in xrange(n_vectors):
            b = _lasvm(self, self.dual_coef_[i],
                       X, Y[:, i], kcache, self.selection, self.search_size,
                       self.termination, self.sv_upper_bound, self.tau,
                       self.finish_step, self.C, self.max_iter, rs,
                       self.callback,
                       verbose=self.verbose, warm_start=warm_start)
            self.intercept_[i] = b

        sv = np.sum(self.dual_coef_ != 0, axis=0, dtype=bool)

        if self.kernel != "precomputed":
            if not self.warm_start:
                self.dual_coef_ = np.ascontiguousarray(self.dual_coef_[:, sv])
                mask = safe_mask(X, sv)
                self.support_vectors_ = X[mask]
            else:
                # Cannot trim the non-zero weights if warm start is used...
                self.support_vectors_ = X

        if self.verbose >= 1:
            print "Number of support vectors:", np.sum(sv)

        return self

    def n_support_vectors(self):
        return np.sum(self.dual_coef_ != 0)

    def decision_function(self, X):
        out = np.zeros((X.shape[0], self.dual_coef_.shape[0]), dtype=np.float64)
        sv = self.support_vectors_ if self.kernel != "precomputed" else X
        decision_function_alpha(X, sv, self.dual_coef_, self._get_kernel(), out)
        return out

    def predict(self, X):
        out = np.zeros(X.shape[0], dtype=np.float64)
        sv = self.support_vectors_ if self.kernel != "precomputed" else X
        predict_alpha(X, sv, self.dual_coef_, self.classes_,
                      self._get_kernel(), out)
        return out
