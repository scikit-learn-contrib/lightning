# Author: Mathieu Blondel
# License: BSD

import numpy as np
import scipy.sparse as sp

from sklearn.base import BaseEstimator as _BaseEstimator
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder

from .random import RandomState


class BaseEstimator(_BaseEstimator):

    def _get_random_state(self):
        return RandomState(seed=self.random_state)

    def n_nonzero(self, percentage=False):
        if hasattr(self, "dual_coef_"):
            coef = self.dual_coef_
        elif hasattr(self, "coef_avg_"):
            coef = self.coef_avg_
        else:
            coef = self.coef_

        n_nz = np.sum(np.sum(coef != 0, axis=0, dtype=bool))

        if percentage:
            if hasattr(self, "support_vectors_") and \
               self.support_vectors_ is not None:
                n_nz /= float(self.n_samples_)
            else:
                n_nz /= float(coef.shape[1])

        return n_nz


class BaseClassifier(BaseEstimator):

    def predict_proba(self, X):
        if len(self.classes_) != 2:
            raise NotImplementedError("predict_(log_)proba only supported"
                                      " for binary classification")

        if self.loss == "log":
            df = self.decision_function(X).ravel()
            prob = 1.0 / (1.0 + np.exp(-df))
        elif self.loss == "modified_huber":
            df = self.decision_function(X).ravel()
            prob = np.minimum(1, np.maximum(-1, df))
            prob += 1
            prob /= 2
        else:
            raise NotImplementedError("predict_(log_)proba only supported when"
                                      " loss='log' or loss='modified_huber' "
                                      "(%s given)" % self.loss)

        out = np.zeros((X.shape[0], 2), dtype=np.float64)
        out[:, 1] = prob
        out[:, 0] = 1 - prob

        return out

    def _set_label_transformers(self, y, reencode=False, neg_label=-1):
        if reencode:
            self.label_encoder_ = LabelEncoder()
            y = self.label_encoder_.fit_transform(y).astype(np.int32)
        else:
            y = y.astype(np.int32)

        self.label_binarizer_ = LabelBinarizer(neg_label=neg_label,
                                               pos_label=1)
        self.label_binarizer_.fit(y)
        self.classes_ = self.label_binarizer_.classes_.astype(np.int32)
        n_classes = len(self.label_binarizer_.classes_)
        n_vectors = 1 if n_classes <= 2 else n_classes

        return y, n_classes, n_vectors

    def decision_function(self, X):
        pred = safe_sparse_dot(X, self.coef_.T)
        if hasattr(self, "intercept_"):
            pred += self.intercept_
        return pred

    def predict(self, X):
        pred = self.decision_function(X)
        out = self.label_binarizer_.inverse_transform(pred)

        if hasattr(self, "label_encoder_"):
            out = self.label_encoder_.inverse_transform(out)

        return out


class BaseRegressor(BaseEstimator):

    def predict(self, X):
        if hasattr(self, "coef_avg_"):
            coef = self.coef_avg_
        else:
            coef = self.coef_

        pred = safe_sparse_dot(X, coef.T)

        if hasattr(self, "intercept_"):
            pred += self.intercept_

        if not self.outputs_2d_:
            pred = pred.ravel()

        return pred
