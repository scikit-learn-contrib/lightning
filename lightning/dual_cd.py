# Author: Mathieu Blondel
# License: BSD

import numpy as np
import scipy.sparse as sp

from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.extmath import safe_sparse_dot

from .base import BaseClassifier
from .dual_cd_fast import _dual_cd

class DualSVC(BaseClassifier, ClassifierMixin):

    def __init__(self, C=1.0, loss="l1", max_iter=1000, tol=1e-3,
                 termination="convergence", n_components=1000,
                 kernel="linear", gamma=0.1, coef0=1, degree=4, cache_mb=500,
                 selection="cyclic", search_size=60, permute=True,
                 shrinking=True, warm_start=False, random_state=None,
                 callback=None, verbose=0, n_jobs=1):
        self.C = C
        self.loss = loss
        self.max_iter = max_iter
        self.tol = tol
        self.termination = termination
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
        self.cache_mb = cache_mb
        self.selection = selection
        self.search_size = search_size
        self.permute = permute
        self.shrinking = shrinking
        self.warm_start = warm_start
        self.random_state = random_state
        self.callback = callback
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.coef_ = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        rs = self._get_random_state()

        self.label_binarizer_ = LabelBinarizer(neg_label=-1, pos_label=1)
        Y = np.asfortranarray(self.label_binarizer_.fit_transform(y),
                              dtype=np.float64)
        n_vectors = Y.shape[1]

        ds = self._get_dataset(X, kernel=False)
        kds = self._get_dataset(X)

        if not self.warm_start or self.coef_ is None:
            self.coef_ = np.zeros((n_vectors, n_features), dtype=np.float64)
            self.dual_coef_ = np.zeros((n_vectors, n_samples), dtype=np.float64)
        self.intercept_ = 0

        for i in xrange(n_vectors):
            _dual_cd(self, self.coef_[i], self.dual_coef_[i],
                     ds, kds, Y[:, i], self.kernel == "linear",
                     self.selection, self.search_size, self.permute,
                     self.termination, self.n_components,
                     self.C, self.loss, self.max_iter, rs, self.tol,
                     self.shrinking, self.callback, verbose=self.verbose)

        if self.kernel != "linear":
            self._post_process_dual(X)

        return self

    def decision_function(self, X):
        if self.kernel == "linear":
            return safe_sparse_dot(X, self.coef_.T) + self.intercept_
        else:
            ds = self._get_dataset(X, self.support_vectors_)
            return ds.dot(self.dual_coef_.T) + self.intercept_

