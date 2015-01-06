# Author: Mathieu Blondel
# License: BSD

import numpy as np

from sklearn.base import ClassifierMixin
from sklearn.utils import check_random_state
from sklearn.preprocessing import LabelBinarizer

from .base import BaseClassifier
from .dataset_fast import get_dataset
from .sdca_fast import _prox_sdca_fit


class SDCAClassifier(BaseClassifier, ClassifierMixin):
    """
    Solves the following objective by (proximal) SDCA:

        minimize_w  1 / n_samples * \sum_i loss(w^T x_i, y_i)
                    + alpha * l1_ratio * ||w||_1
                    + alpha * (1 - l1_ratio) * 0.5 * ||w||^2_2
    """

    def __init__(self, alpha=1.0, l1_ratio=0, loss="hinge", gamma=1.0,
                 max_iter=100, tol=1e-3, callback=None, n_calls=None, verbose=0,
                 random_state=None):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.loss = loss
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol
        self.callback = callback
        self.n_calls = n_calls
        self.verbose = verbose
        self.random_state = random_state

    def _get_loss(self):
        losses = {
            "squared": 0,
            "absolute": 1,
            "hinge": 2,
            "smooth_hinge": 3,
            "squared_hinge": 4,
        }
        return losses[self.loss]

    def _get_alpha2_lasso(self, y, alpha1):
        if self.loss == "squared":
            y_bar = 0.5 * np.mean(y ** 2)

        elif self.loss == "absolute":
            y_bar = np.mean(np.abs(y))

        elif self.loss in ("hinge", "squared_hinge"):
            y_bar = 1.0

        elif self.loss == "smooth_hinge":
            if self.gamma < 1:
                y_bar = 1 - 0.5 * self.gamma
            else:
                y_bar = 0.5 / self.gamma

        else:
            raise ValueError("Unknown loss.")

        return  self.tol * (alpha1 / y_bar) ** 2


    def fit(self, X, y):
        n_samples, n_features = X.shape

        rng = check_random_state(self.random_state)
        loss = self._get_loss()

        self.label_binarizer_ = LabelBinarizer(neg_label=-1, pos_label=1)
        Y = np.asfortranarray(self.label_binarizer_.fit_transform(y),
                              dtype=np.float64)
        n_vectors = Y.shape[1]

        ds = get_dataset(X, order="c")

        self.coef_ = np.zeros((n_vectors, n_features), dtype=np.float64)
        self.dual_coef_ = np.zeros((n_vectors, n_samples), dtype=np.float64)

        alpha1 = self.l1_ratio * self.alpha
        alpha2 = (1 - self.l1_ratio) * self.alpha

        if self.loss == "squared_hinge":
            # For consistency with the rest of lightning.
            alpha1 *= 0.5
            alpha2 *= 0.5

        tol = self.tol
        n_calls = n_samples if self.n_calls is None else self.n_calls

        for i in xrange(n_vectors):
            y = Y[:, i]

            if self.l1_ratio == 1.0:
                # Prox-SDCA needs a strongly convex regularizer so adds some
                # L2 penalty (see paper).
                alpha2 = self._get_alpha2_lasso(y, alpha1)
                tol = self.tol * 0.5

            _prox_sdca_fit(self, ds, y, self.coef_[i], self.dual_coef_[i],
                           alpha1, alpha2, loss, self.gamma, self.max_iter,
                           tol, self.callback, n_calls, self.verbose, rng)

        return self
