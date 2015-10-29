# Author: Mathieu Blondel
# License: BSD

import numpy as np

from sklearn.preprocessing import LabelBinarizer
from sklearn.externals.six.moves import xrange

from .base import BaseClassifier, BaseRegressor
from .dataset_fast import get_dataset
from .sag_fast import _sag_fit

from .sgd_fast import ModifiedHuber
from .sgd_fast import SmoothHinge
from .sgd_fast import SquaredHinge
from .sgd_fast import Log
from .sgd_fast import SquaredLoss


class _BaseSAG(object):

    def _finalize_coef(self):
        self.coef_ *= self.coef_scale_
        self.coef_scale_.fill(1.0)

    def _fit(self, X, Y):
        n_samples, n_features = X.shape
        rng = self._get_random_state()
        loss = self._get_loss()
        n_vectors = Y.shape[1]
        n_inner = int(self.n_inner * n_samples)
        ds = get_dataset(X, order="c")

        self.coef_ = np.zeros((n_vectors, n_features), dtype=np.float64)
        self.coef_scale_ = np.ones(n_vectors, dtype=np.float64)
        grad = np.zeros((n_vectors, n_samples), dtype=np.float64)

        for i in xrange(n_vectors):
            y = Y[:, i]

            _sag_fit(self, ds, y, self.coef_[i], self.coef_scale_[i:], grad[i],
                     self.eta, self.alpha, loss, self.max_iter, n_inner,
                     self.tol, self.verbose, self.callback, rng, self.is_saga)

        return self


class SAGClassifier(BaseClassifier, _BaseSAG):
    """
    Estimator for learning linear classifiers by SAG.

    Solves the following objective:

        minimize_w  1 / n_samples * \sum_i loss(w^T x_i, y_i)
                    + alpha * 0.5 * ||w||^2_2
    """

    def __init__(self, eta=1.0, alpha=1.0, loss="smooth_hinge", gamma=1.0,
                 max_iter=10, n_inner=1.0, tol=1e-3, verbose=0,
                 callback=None, random_state=None):
        self.eta = eta
        self.alpha = alpha
        self.loss = loss
        self.gamma = gamma
        self.max_iter = max_iter
        self.n_inner = n_inner
        self.tol = tol
        self.verbose = verbose
        self.callback = callback
        self.random_state = random_state
        self.is_saga = False

    def _get_loss(self):
        losses = {
            "modified_huber": ModifiedHuber(),
            "smooth_hinge": SmoothHinge(self.gamma),
            "squared_hinge": SquaredHinge(1.0),
            "log": Log(),
            "squared": SquaredLoss(),
        }
        return losses[self.loss]

    def fit(self, X, y):
        self.label_binarizer_ = LabelBinarizer(neg_label=-1, pos_label=1)
        Y = np.asfortranarray(self.label_binarizer_.fit_transform(y),
                              dtype=np.float64)
        return self._fit(X, Y)


class SAGAClassifier(SAGClassifier):
    """
    Estimator for learning linear classifiers by SAGA.

    Solves the following objective:

        minimize_w  1 / n_samples * \sum_i loss(w^T x_i, y_i)
                    + alpha * 0.5 * ||w||^2_2

    """

    def __init__(self, eta=1.0, alpha=1.0, loss="smooth_hinge", gamma=1.0,
                 max_iter=10, n_inner=1.0, tol=1e-3, verbose=0,
                 callback=None, random_state=None):
            super(SAGAClassifier, self).__init__(
                eta=eta, alpha=alpha, loss=loss, gamma=gamma,
                max_iter=max_iter, n_inner=n_inner, tol=tol, verbose=verbose,
                callback=callback, random_state=random_state)
            self.is_saga = True


class SAGRegressor(BaseRegressor, _BaseSAG):
    """
    Estimator for learning linear regressors by SAG.

    Solves the following objective:

        minimize_w  1 / n_samples * \sum_i loss(w^T x_i, y_i)
                    + alpha * 0.5 * ||w||^2_2
    """

    def __init__(self, eta=1.0, alpha=1.0, loss="smooth_hinge", gamma=1.0,
                 max_iter=10, n_inner=1.0, tol=1e-3, verbose=0,
                 callback=None, random_state=None):
        self.eta = eta
        self.alpha = alpha
        self.loss = loss
        self.gamma = gamma
        self.max_iter = max_iter
        self.n_inner = n_inner
        self.tol = tol
        self.verbose = verbose
        self.callback = callback
        self.random_state = random_state
        self.is_saga = False

    def _get_loss(self):
        losses = {
            "modified_huber": ModifiedHuber(),
            "smooth_hinge": SmoothHinge(self.gamma),
            "squared_hinge": SquaredHinge(1.0),
            "log": Log(),
            "squared": SquaredLoss(),
        }
        return losses[self.loss]

    def fit(self, X, y):
        self.outputs_2d_ = len(y.shape) > 1
        Y = y.reshape(-1, 1) if not self.outputs_2d_ else y
        Y = Y.astype(np.float64)
        return self._fit(X, Y)


class SAGARegressor(SAGRegressor):
    """
    Estimator for learning linear regressors by SAG.

    Solves the following objective:

        minimize_w  1 / n_samples * \sum_i loss(w^T x_i, y_i)
                    + alpha * 0.5 * ||w||^2_2
    """

    def __init__(self, eta=1.0, alpha=1.0, loss="smooth_hinge", gamma=1.0,
                 max_iter=10, n_inner=1.0, tol=1e-3, verbose=0,
                 callback=None, random_state=None):
            super(SAGARegressor, self).__init__(
                eta=eta, alpha=alpha, loss=loss, gamma=gamma,
                max_iter=max_iter, n_inner=n_inner, tol=tol, verbose=verbose,
                callback=callback, random_state=random_state)
            self.is_saga = True
