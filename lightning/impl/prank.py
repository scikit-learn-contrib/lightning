# Author: Mathieu Blondel
# License: BSD

import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.metrics.pairwise import pairwise_kernels

from .base import BaseEstimator
from .dataset_fast import get_dataset
from .prank_fast import _prank_fit
from .prank_fast import _prank_fit_kernel
from .prank_fast import _prank_predict


class _BasePRank(BaseEstimator):

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(np.abs(y - y_pred))

    @property
    def classes_(self):
        return self._label_encoder.classes_


class PRank(_BasePRank):

    def __init__(self, n_iter=10, shuffle=True, random_state=None):
        self.n_iter = n_iter
        self.shuffle = shuffle
        self.random_state = random_state

    def fit(self, X, y):
        n_samples, n_features = X.shape
        rs = self._get_random_state()

        self._label_encoder = LabelEncoder()
        y = self._label_encoder.fit_transform(y).astype(np.int32)
        n_classes = len(self.classes_)


        self.coef_ = np.zeros(n_features, dtype=np.float64)
        self.thresholds_ = np.zeros(n_classes, dtype=np.float64)
        self.thresholds_[-1] = np.inf

        ds = get_dataset(X)

        _prank_fit(self.coef_, self.thresholds_, ds, y, n_classes,
                   self.n_iter, rs, self.shuffle)

        return self

    def predict(self, X):
        n_samples = X.shape[0]
        dot = safe_sparse_dot(X, self.coef_)
        out = np.zeros(n_samples, dtype=np.int32)
        _prank_predict(dot, self.thresholds_, len(self.classes_), out)
        return self._label_encoder.inverse_transform(out)


class KernelPRank(_BasePRank):

    def __init__(self, n_iter=10, shuffle=True, random_state=None,
                 kernel="linear", gamma=None, degree=3, coef0=1,
                 kernel_params=None, ):
        self.n_iter = n_iter
        self.shuffle = shuffle
        self.random_state = random_state
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params

    @property
    def _pairwise(self):
        return self.kernel == "precomputed"

    def _get_kernel(self, X, Y=None):
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            params = {"gamma": self.gamma,
                      "degree": self.degree,
                      "coef0": self.coef0}
        return pairwise_kernels(X, Y, metric=self.kernel,
                                filter_params=True, **params)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        rs = self._get_random_state()

        self._label_encoder = LabelEncoder()
        y = self._label_encoder.fit_transform(y).astype(np.int32)
        n_classes = len(self.classes_)

        K = self._get_kernel(X)
        self.X_train_ = X

        self.dual_coef_ = np.zeros(n_samples, dtype=np.float64)
        self.thresholds_ = np.zeros(n_classes, dtype=np.float64)
        self.thresholds_[-1] = np.inf

        _prank_fit_kernel(self.dual_coef_, self.thresholds_, K, y, n_classes,
                   self.n_iter, rs, self.shuffle)

        return self

    def predict(self, X):
        K = self._get_kernel(X, self.X_train_)
        n_samples = X.shape[0]
        dot = np.dot(K, self.dual_coef_)
        out = np.zeros(n_samples, dtype=np.int32)
        _prank_predict(dot, self.thresholds_, len(self.classes_), out)
        return self._label_encoder.inverse_transform(out)
