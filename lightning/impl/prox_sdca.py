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

        for i in xrange(n_vectors):
            _prox_sdca_fit(self, ds, Y[:, i], self.coef_[i], self.dual_coef_[i],
                           self.alpha, self.l1_ratio, loss, self.max_iter,
                           self.tol, self.callback, self.verbose, rng)

        return self
