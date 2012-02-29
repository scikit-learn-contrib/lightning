# Author: Mathieu Blondel
# License: BSD

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_random_state
from sklearn.metrics.pairwise import pairwise_kernels

from dual_cd_fast import _dual_cd

class DualLinearSVC(BaseEstimator):

    def __init__(self, C=1.0, loss="l1", max_iter=1000,
                 random_state=None, verbose=0, n_jobs=1):
        self.C = C
        self.loss = loss
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose
        self.n_jobs = n_jobs

    def fit(self, X, y):
        rs = check_random_state(self.random_state)
        self.label_binarizer_ = LabelBinarizer(neg_label=-1, pos_label=1)
        Y = self.label_binarizer_.fit_transform(y)
        W = [_dual_cd(X, Y[:, i],
                      self.C, self.loss, self.max_iter, rs,
                      precomputed_kernel=False, verbose=self.verbose) \
                for i in range(Y.shape[1])]
        self.coef_ = np.array(W)

    def decision_function(self, X):
        return np.dot(X, self.coef_.T)

    def predict(self, X):
        pred = self.decision_function(X)
        return self.label_binarizer_.inverse_transform(pred, threshold=0)


class DualSVC(BaseEstimator):

    def __init__(self, C=1.0, loss="l1", max_iter=1000, tol=1e-4,
                 kernel="linear", gamma=0.1, coef0=1, degree=4,
                 random_state=None, verbose=0, n_jobs=1):
        self.C = C
        self.loss = loss
        self.kernel = kernel
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose
        self.n_jobs = n_jobs

    def _kernel_params(self):
        return {"gamma" : self.gamma,
                "degree" : self.degree,
                "coef0" : self.coef0}

    def fit(self, X, y):
        rs = check_random_state(self.random_state)
        self.label_binarizer_ = LabelBinarizer(neg_label=-1, pos_label=1)
        Y = self.label_binarizer_.fit_transform(y)
        K = pairwise_kernels(X, X, metric=self.kernel,
                             filter_params=True, n_jobs=self.n_jobs,
                             **self._kernel_params())
        Alpha = [_dual_cd(K, Y[:, i],
                          self.C, self.loss, self.max_iter, rs,
                          precomputed_kernel=True, verbose=self.verbose)
                    for i in range(Y.shape[1])]
        self.dual_coef_ = np.array(Alpha) * Y.T
        # FIXME: can trim the model
        self.X_train_ = X

    def decision_function(self, X):
        K = np.dot(X, self.X_train_.T)
        return np.dot(K, self.dual_coef_.T)

    def predict(self, X):
        pred = self.decision_function(X)
        return self.label_binarizer_.inverse_transform(pred, threshold=0)
