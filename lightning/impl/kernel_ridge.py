import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.linear_model.ridge import _solve_dense_cholesky_kernel


class KernelRidge(BaseEstimator, RegressorMixin):

    def __init__(self, alpha=1, kernel="linear", gamma=None, degree=3, coef0=1,
                 kernel_params=None):
        self.alpha = alpha
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params

    def _get_kernel(self, X, Y=None):
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            params = {"gamma": self.gamma,
                      "degree": self.degree,
                      "coef0": self.coef0}
        return pairwise_kernels(X, Y, metric=self.kernel,
                                filter_params=True, **params)

    @property
    def _pairwise(self):
        return self.kernel == "precomputed"

    def fit(self, X, y=None, sample_weight=1.0):
        n_samples = X.shape[0]
        K = self._get_kernel(X)
        alpha = np.array([self.alpha])

        ravel = False
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
            ravel = True


        self.dual_coef_ = _solve_dense_cholesky_kernel(K, y, alpha,
                                                       sample_weight)
        if ravel:
            self.dual_coef_ = self.dual_coef_.ravel()

        self.X_fit_ = X

        return self

    def predict(self, X, ret_variance=False):
        K = self._get_kernel(X, self.X_fit_)
        return np.dot(K, self.dual_coef_)
