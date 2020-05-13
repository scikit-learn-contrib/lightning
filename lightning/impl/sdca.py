# Author: Mathieu Blondel
# License: BSD

import numpy as np

from sklearn.utils import check_random_state
from sklearn.preprocessing import LabelBinarizer
from six.moves import xrange

from .base import BaseClassifier, BaseRegressor
from .dataset_fast import get_dataset
from .sdca_fast import _prox_sdca_fit


class _BaseSDCA(object):

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

    def _fit(self, X, Y):
        n_samples, n_features = X.shape
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
        rng = check_random_state(self.random_state)
        loss = self._get_loss()

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


class SDCAClassifier(BaseClassifier, _BaseSDCA):
    """
    Estimator for learning linear classifiers by (proximal) SDCA.

    Solves the following objective:

        minimize_w  1 / n_samples * \sum_i loss(w^T x_i, y_i)
                    + alpha * l1_ratio * ||w||_1
                    + alpha * (1 - l1_ratio) * 0.5 * ||w||^2_2

    Parameters
    ----------
    loss: string, {'squared', 'absolute', 'hinge', 'smooth_hinge', 'squared_hinge'}
         Loss function to use in the model.
    alpha: float
         Amount of regularization (see model formulation above).
    l1_ratio: float
         Ratio between the L1 and L2 regularization (see model formulation above).
    gamma : float
        gamma parameter in the "smooth_hinge" loss (not used for other
        loss functions)
    tol : float
        stopping criterion tolerance.
    max_iter : int
        maximum number of outer iterations (also known as epochs).
    verbose : int
        verbosity level. Set positive to print progress information.
    callback : callable or None
        if given, callback(self) will be called on each outer iteration
        (epoch).
    random_state: int or RandomState
        Pseudo-random number generator state used for random sampling.
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

    def fit(self, X, y):
        self._set_label_transformers(y)
        Y = np.asfortranarray(self.label_binarizer_.transform(y),
                              dtype=np.float64)
        return self._fit(X, Y)


class SDCARegressor(BaseRegressor, _BaseSDCA):
    """
    Estimator for learning linear regressors by (proximal) SDCA.

    Solves the following objective:

        minimize_w  1 / n_samples * \sum_i loss(w^T x_i, y_i)
                    + alpha * l1_ratio * ||w||_1
                    + alpha * (1 - l1_ratio) * 0.5 * ||w||^2_2
    Parameters
    ----------
    loss: string, {'squared', 'absolute'}
         Loss function to use in the model.
    alpha: float
         Amount of regularization (see model formulation above).
    l1_ratio: float
         Ratio between the L1 and L2 regularization (see model formulation above).
    tol : float
        stopping criterion tolerance.
    max_iter : int
        maximum number of outer iterations (also known as epochs).
    verbose : int
        verbosity level. Set positive to print progress information.
    callback : callable or None
        if given, callback(self) will be called on each outer iteration
        (epoch).
    random_state: int or RandomState
        Pseudo-random number generator state used for random sampling.
    """

    def __init__(self, alpha=1.0, l1_ratio=0, loss="squared",
                 max_iter=100, tol=1e-3, callback=None, n_calls=None, verbose=0,
                 random_state=None):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.loss = loss
        self.gamma = 1.0
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
        }
        return losses[self.loss]

    def fit(self, X, y):
        self.outputs_2d_ = len(y.shape) > 1
        Y = y.reshape(-1, 1) if not self.outputs_2d_ else y
        Y = Y.astype(np.float64)
        return self._fit(X, Y)
