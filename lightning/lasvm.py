# Author: Mathieu Blondel
# License: BSD

import numpy as np

from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.extmath import safe_sparse_dot

from .base import BaseClassifier
from .lasvm_fast import _lasvm

class LaSVM(BaseClassifier, ClassifierMixin):

    def __init__(self, C=1.0, max_iter=10,
                 kernel="linear", gamma=0.1, coef0=1, degree=4,
                 selection="cyclic", search_size=60, permute=True,
                 termination="n_iter", n_components=1000,
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
        self.permute = permute
        self.termination = termination
        self.n_components = n_components
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

    def fit(self, X, y):
        n_samples = X.shape[0]
        rs = self._get_random_state()

        self.label_binarizer_ = LabelBinarizer(neg_label=-1, pos_label=1)
        Y = np.asfortranarray(self.label_binarizer_.fit_transform(y),
                              dtype=np.float64)
        self.classes_ = self.label_binarizer_.classes_.astype(np.int32)
        n_vectors = Y.shape[1]

        dual_coef = np.zeros((n_vectors, n_samples), dtype=np.float64)

        warm_start = False
        if self.warm_start and self.dual_coef_ is not None:
            warm_start = True
            dual_coef[:, self.support_indices_] = self.dual_coef_
        else:
            self.intercept_ = np.zeros((n_vectors,), dtype=np.float64)

        self.dual_coef_ = dual_coef
        ds = self._get_dataset(X)

        for i in xrange(n_vectors):
            b = _lasvm(self, self.dual_coef_[i], ds, Y[:, i],
                       self.selection, self.search_size, self.permute,
                       self.termination, self.n_components, self.tau,
                       self.finish_step, self.C, self.max_iter, rs,
                       self.callback, verbose=self.verbose,
                       warm_start=warm_start)
            self.intercept_[i] = b

        if self.kernel == "linear":
            self.coef_ = safe_sparse_dot(self.dual_coef_, X)

        self._post_process_dual(X)

        return self

    def decision_function(self, X):
        if self.kernel == "linear":
            return safe_sparse_dot(X, self.coef_.T) + self.intercept_
        else:
            ds = self._get_dataset(X, self.support_vectors_)
            return ds.dot(self.dual_coef_.T) + self.intercept_

