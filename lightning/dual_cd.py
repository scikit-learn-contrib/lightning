# Author: Mathieu Blondel
# License: BSD

import numpy as np
import scipy.sparse as sp

from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.extmath import safe_sparse_dot

from .base import BaseClassifier
from .dataset_fast import get_dataset
from .dual_cd_fast import _dual_cd


class LinearSVC(BaseClassifier, ClassifierMixin):

    def __init__(self, C=1.0, loss="l1", max_iter=1000, tol=1e-3,
                 permute=True, shrinking=True, warm_start=False,
                 random_state=None, callback=None, verbose=0, n_jobs=1):
        self.C = C
        self.loss = loss
        self.max_iter = max_iter
        self.tol = tol
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

        ds = get_dataset(X)

        if not self.warm_start or self.coef_ is None:
            self.coef_ = np.zeros((n_vectors, n_features), dtype=np.float64)
            self.dual_coef_ = np.zeros((n_vectors, n_samples), dtype=np.float64)
        self.intercept_ = 0

        for i in xrange(n_vectors):
            _dual_cd(self, self.coef_[i], self.dual_coef_[i],
                     ds, Y[:, i], self.permute,
                     self.C, self.loss, self.max_iter, rs, self.tol,
                     self.shrinking, self.callback, verbose=self.verbose)

        return self
