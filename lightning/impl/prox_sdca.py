import numpy as np

from sklearn.base import ClassifierMixin
from sklearn.utils import check_random_state
from sklearn.preprocessing import LabelBinarizer

from .base import BaseClassifier
from .dataset_fast import get_dataset
from .prox_sdca_fast import _prox_sdca_fit


class ProxSDCA_Classifier(BaseClassifier, ClassifierMixin):
    """
    Solves the following objective by ProxSDCA:

        minimize_w  1 / n_samples * \sum_i loss(w^T x_i, y_i)
                    + alpha * l1_ratio * ||w||_1
                    + alpha * (1 - l1_ratio) * 0.5 * ||w||^2_2
    """

    def __init__(self, alpha=1.0, l1_ratio=0, loss="hinge", max_iter=100,
                 tol=1e-3, callback=None, verbose=0, random_state=None):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.loss = loss
        self.max_iter = max_iter
        self.tol = tol
        self.callback = callback
        self.verbose = verbose
        self.random_state = random_state

    def _get_loss(self):
        losses = {
            "squared": 0,
            "absolute": 1,
            "hinge": 2,
        }
        return losses[self.loss]

    def _get_alpha2_lasso(self, y, alpha1):
        if self.loss == "squared":
            y_bar = 0.5 * np.mean(y ** 2)
        elif self.loss == "absolute":
            y_bar = np.mean(np.abs(y))
        elif self.loss == "hinge":
            y_bar = 1.0 / y.shape[0]
        else:
            raise ValueError("Unknown loss.")
        return  self.tol * alpha1 ** 2 / (y_bar ** 2)


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
        tol = self.tol

        for i in xrange(n_vectors):
            y = Y[:, i]

            if self.l1_ratio == 1.0:
                # ProxSDCA needs a strongly convex regularizer so adds some
                # L2 penalty (see paper).
                alpha2 = self._get_alpha2_lasso(y, alpha1)
                tol = self.tol * 0.5

            _prox_sdca_fit(self, ds, y, self.coef_[i], self.dual_coef_[i],
                           alpha1, alpha2, loss, self.max_iter,
                           tol, self.callback, self.verbose, rng)

        return self
