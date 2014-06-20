# Author: Mathieu Blondel
# License: BSD

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder

class _BasePRank(BaseEstimator):

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(np.abs(y - y_pred))


class PRank(_BasePRank):

    def __init__(self, n_iter=10):
        self.n_iter = n_iter

    def _predit(self, x):
        pred = np.dot(x, self.coef_)
        n_classes = len(self.thresholds_)
        y_hat = 0

        for r in xrange(n_classes):
            if pred - self.thresholds_[r] < 0:
                y_hat = r
                break

        return y_hat, pred

    @property
    def classes_(self):
        return self.label_encoder_.classes_

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.label_encoder_ = LabelEncoder()
        y = self.label_encoder_.fit_transform(y)
        n_classes = len(self.classes_)

        self.coef_ = np.zeros(n_features, dtype=np.float64)
        self.thresholds_ = np.zeros(n_classes, dtype=np.float64)
        self.thresholds_[-1] = np.inf
        yr = np.ones(n_classes)

        for n in xrange(self.n_iter):
            for i in xrange(n_samples):

                y_hat, pred = self._predit(X[i])

                if y_hat == y[i]:
                    continue

                for r in xrange(n_classes - 1):
                    if y[i] <= r:
                        yr[r] = -1
                    else:
                        yr[r] = 1

                tau = 0
                for r in xrange(n_classes - 1):
                    if yr[r] * (pred - self.thresholds_[r]) <= 0:
                        tau += yr[r]
                        self.thresholds_[r] -= yr[r]

                self.coef_ += tau * X[i]

        return self

    def predict(self, X):
        n_samples = X.shape[0]
        pred = np.zeros(X.shape[0], dtype=int)
        for i in xrange(n_samples):
            pred[i] = self._predit(X[i])[0]
        return self.label_encoder_.inverse_transform(pred)
