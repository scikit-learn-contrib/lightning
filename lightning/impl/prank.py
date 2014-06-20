# Author: Mathieu Blondel
# License: BSD

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.extmath import safe_sparse_dot

from .dataset_fast import get_dataset
from .prank_fast import _prank_fit
from .prank_fast import _prank_predict


class _BasePRank(BaseEstimator):

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(np.abs(y - y_pred))


class PRank(_BasePRank):

    def __init__(self, n_iter=10):
        self.n_iter = n_iter

    @property
    def classes_(self):
        return self.label_encoder_.classes_

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.label_encoder_ = LabelEncoder()
        y = self.label_encoder_.fit_transform(y).astype(np.int32)
        n_classes = len(self.classes_)

        self.coef_ = np.zeros(n_features, dtype=np.float64)
        self.thresholds_ = np.zeros(n_classes, dtype=np.float64)
        self.thresholds_[-1] = np.inf

        ds = get_dataset(X)

        _prank_fit(self.coef_, self.thresholds_, ds, y, n_classes, self.n_iter)

        return self

    def predict(self, X):
        n_samples = X.shape[0]
        dot = safe_sparse_dot(X, self.coef_)
        out = np.zeros(n_samples, dtype=np.int32)
        _prank_predict(dot, self.thresholds_, len(self.classes_), out)
        return self.label_encoder_.inverse_transform(out)
