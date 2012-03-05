# Author: Mathieu Blondel
# License: BSD

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_random_state, safe_mask
from sklearn.metrics.pairwise import pairwise_kernels

from primal_cd_fast import _primal_cd_l2svm_l1r
from primal_cd_fast import _primal_cd_l2svm_l2r

class PrimalLinearSVC(BaseEstimator, ClassifierMixin):

    def __init__(self, C=1.0, penalty="l2", max_iter=1000, tol=1e-4,
                 warm_start=False, random_state=None, verbose=0, n_jobs=1):
        self.C = C
        self.penalty = penalty
        self.max_iter = max_iter
        self.tol = tol
        self.warm_start = warm_start
        self.random_state = random_state
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.coef_ = None

    def fit(self, X, y):
        n_features = X.shape[1]
        rs = check_random_state(self.random_state)
        self.label_binarizer_ = LabelBinarizer(neg_label=-1, pos_label=1)
        Y = self.label_binarizer_.fit_transform(y)
        n_vectors = Y.shape[1]

        if not self.warm_start or self.coef_ is None:
            self.coef_ = np.zeros((n_vectors, n_features), dtype=np.float64)

        if self.penalty == "l1":
            func = _primal_cd_l2svm_l1r
        else:
            func = _primal_cd_l2svm_l2r

        for i in xrange(n_vectors):
            func(self.coef_[i], X, Y[:, i],
                 self.C, self.max_iter, rs, self.tol,
                 verbose=self.verbose)

        return self

    def decision_function(self, X):
        return np.dot(X, self.coef_.T)

    def predict(self, X):
        pred = self.decision_function(X)
        return self.label_binarizer_.inverse_transform(pred, threshold=0)


