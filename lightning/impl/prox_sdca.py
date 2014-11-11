import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state
from sklearn.utils.extmath import safe_sparse_dot

from .dataset_fast import get_dataset
from .prox_sdca_fast import _prox_sdca_fit


class ProxSDCA_Classifier(BaseEstimator, ClassifierMixin):

    def __init__(self, alpha=1.0, l1_ratio=0, loss="hinge", max_iter=10,
                 tol=1e-3, random_state=None):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.loss = loss
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _get_loss(self):
        losses = {
            "squared": 0,
            "absolute": 1,
            "hinge": 2,
        }
        return losses[self.loss]

    def fit(self, X, y):
        n_samples, n_features = X.shape

        rng = check_random_state(self.random_state)
        loss = self._get_loss()

        ds = get_dataset(X, order="c")
        y = np.array(y, dtype=np.float64)

        self.coef_ = np.zeros(n_features, dtype=np.float64)
        self.dual_coef_ = np.zeros(n_samples, dtype=np.float64)

        _prox_sdca_fit(ds, y, self.coef_, self.dual_coef_,
                       self.alpha, self.l1_ratio, loss, self.max_iter, self.tol,
                       rng)

        return self

    def predict(self, X):
        return np.sign(safe_sparse_dot(X, self.coef_))
