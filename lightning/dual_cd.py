# Author: Mathieu Blondel
# License: BSD

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_random_state
from sklearn.metrics.pairwise import pairwise_kernels

from .base import BaseLinearClassifier, BaseKernelClassifier
from .dual_cd_fast import _dual_cd
from .kernel_fast import get_kernel, KernelCache

class DualLinearSVC(BaseLinearClassifier, ClassifierMixin):

    def __init__(self, C=1.0, loss="l1", max_iter=1000, tol=1e-3,
                 termination="convergence", n_components=1000,
                 shrinking=True, warm_start=False, random_state=None,
                 callback=None,
                 verbose=0, n_jobs=1):
        self.C = C
        self.loss = loss
        self.max_iter = max_iter
        self.tol = tol
        self.termination = termination
        self.n_components = n_components
        self.shrinking = shrinking
        self.warm_start = warm_start
        self.random_state = random_state
        self.callback = callback
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.coef_ = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        rs = check_random_state(self.random_state)
        self.label_binarizer_ = LabelBinarizer(neg_label=-1, pos_label=1)
        Y = self.label_binarizer_.fit_transform(y)
        n_vectors = Y.shape[1]

        if not self.warm_start or self.coef_ is None:
            self.coef_ = np.zeros((n_vectors, n_features),
                                  dtype=np.float64)
            self.dual_coef_ = np.zeros((n_vectors, n_samples),
                                       dtype=np.float64)
        self.intercept_ = 0

        kernel = get_kernel("linear")
        kcache = KernelCache(kernel, n_samples, 0, 0, self.verbose)

        for i in xrange(n_vectors):
            _dual_cd(self, self.coef_[i], self.dual_coef_[i],
                     X, Y[:, i], kcache, True,
                     "permute", 60, self.termination, self.n_components,
                     self.C, self.loss, self.max_iter, rs, self.tol,
                     self.shrinking, self.callback, verbose=self.verbose)

        return self


class DualSVC(BaseKernelClassifier, ClassifierMixin):

    def __init__(self, C=1.0, loss="l1", max_iter=10, tol=1e-3,
                 shrinking=True, kernel="linear", gamma=0.1, coef0=1, degree=4,
                 selection="permute", search_size=60,
                 termination="convergence", n_components=1000,
                 warm_start=False, random_state=None, cache_mb=500,
                 callback=None,
                 verbose=0, n_jobs=1):
        self.C = C
        self.loss = loss
        self.max_iter = max_iter
        self.tol = tol
        self.shrinking = shrinking
        self.kernel = kernel
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
        self.selection = selection
        self.search_size = search_size
        self.termination = termination
        self.n_components = n_components
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

        if self.warm_start and self.coef_ is not None:
            coef[:, self.support_indices_] = self.coef_

        self.coef_ = coef

        coef = np.empty(0, dtype=np.float64)

        kernel = self._get_kernel()
        kcache = KernelCache(kernel, n_samples, self.cache_mb, 1, self.verbose)

        self.support_vectors_ = X
        self.intercept_ = np.zeros(n_vectors, dtype=np.float64)

        for i in xrange(n_vectors):
            _dual_cd(self, coef, self.coef_[i],
                     X, Y[:, i], kcache, False,
                     self.selection, self.search_size,
                     self.termination, self.n_components,
                     self.C, self.loss, self.max_iter, rs, self.tol,
                     self.shrinking, self.callback, verbose=self.verbose)

        sv = np.sum(self.coef_ != 0, axis=0, dtype=bool)
        self.support_indices_ = np.arange(n_samples)[sv]

        self._post_process(X)

        return self
