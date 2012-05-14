# Author: Mathieu Blondel
# License: BSD

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_random_state, safe_mask
from sklearn.metrics.pairwise import pairwise_kernels

from .base import BaseKernelClassifier
from .kernel_fast import KernelCache
from .lasvm_fast import _lasvm

class LaSVM(BaseKernelClassifier, ClassifierMixin):

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
        self.coef_ = None

    def fit(self, X, y):
        n_samples = X.shape[0]
        rs = check_random_state(self.random_state)

        self.label_binarizer_ = LabelBinarizer(neg_label=-1, pos_label=1)
        Y = self.label_binarizer_.fit_transform(y)
        self.classes_ = self.label_binarizer_.classes_.astype(np.int32)
        n_vectors = Y.shape[1]

        coef = np.zeros((n_vectors, n_samples), dtype=np.float64)

        warm_start = False
        if self.warm_start and self.coef_ is not None:
            warm_start = True
            coef[:, self.support_indices_] = self.coef_
        else:
            self.intercept_ = np.zeros((n_vectors,), dtype=np.float64)

        self.coef_ = coef

        kernel = self._get_kernel()
        kcache = KernelCache(kernel, n_samples, self.cache_mb, 1, self.verbose)
        self.support_vectors_ = X

        for i in xrange(n_vectors):
            b = _lasvm(self, self.coef_[i],
                       X, Y[:, i], kcache, self.selection, self.search_size,
                       self.termination, self.sv_upper_bound, self.tau,
                       self.finish_step, self.C, self.max_iter, rs,
                       self.callback,
                       verbose=self.verbose, warm_start=warm_start)
            self.intercept_[i] = b

        sv = np.sum(self.coef_ != 0, axis=0, dtype=bool)
        self.support_indices_ = np.arange(n_samples)[sv]

        self._post_process(X)

        return self
