# Author: Mathieu Blondel
# License: BSD

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.utils import safe_mask

from .predict_fast import predict_alpha, decision_function_alpha
from .kernel_fast import get_kernel


class BaseLinearClassifier(BaseEstimator):

    def decision_function(self, X):
        return np.dot(X, self.coef_.T) + self.intercept_

    def predict(self, X):
        pred = self.decision_function(X)
        return self.label_binarizer_.inverse_transform(pred, threshold=0)


class BaseKernelClassifier(BaseEstimator):

    def n_support_vectors(self):
        if self.coef_ is None:
            return 0
        else:
            return np.sum(np.sum(self.coef_ != 0, axis=0, dtype=bool))

    def decision_function(self, X):
        out = np.zeros((X.shape[0], self.coef_.shape[0]), dtype=np.float64)

        if self.coef_ is not None:
            sv = self.support_vectors_ if self.kernel != "precomputed" else X
            decision_function_alpha(X, sv, self.coef_, self.intercept_,
                                    self._get_kernel(), out)

        return out

    def predict(self, X):
        out = np.zeros(X.shape[0], dtype=np.float64)

        if self.coef_ is not None:
            sv = self.support_vectors_ if self.kernel != "precomputed" else X
            predict_alpha(X, sv, self.coef_, self.intercept_,
                          self.classes_, self._get_kernel(), out)

        if hasattr(self, "label_encoder_"):
            out = self.label_encoder_.inverse_transform(out)

        return out

    def _kernel_params(self):
        return {"gamma" : self.gamma,
                "degree" : self.degree,
                "coef0" : self.coef0}

    def _get_kernel(self):
        return get_kernel(self.kernel, **self._kernel_params())

    def _post_process(self, X):
        # We can't know the support vectors when using precomputed kernels.
        if self.kernel != "precomputed":
            sv = np.sum(self.coef_ != 0, axis=0, dtype=bool)
            self.coef_ = np.ascontiguousarray(self.coef_[:, sv])
            mask = safe_mask(X, sv)
            self.support_vectors_ = X[mask]

        if self.verbose >= 1:
            print "Number of support vectors:", np.sum(sv)
