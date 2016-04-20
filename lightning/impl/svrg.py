# Author: Mathieu Blondel
# License: BSD

import numpy as np

from sklearn.preprocessing import LabelBinarizer
from sklearn.externals.six.moves import xrange

from .base import BaseClassifier, BaseRegressor
from .dataset_fast import get_dataset
from .svrg_fast import _svrg_fit

from .sgd_fast import ModifiedHuber
from .sgd_fast import SmoothHinge
from .sgd_fast import SquaredHinge
from .sgd_fast import Log
from .sgd_fast import SquaredLoss
from .sgd_fast import WeightedLoss


class _BaseSVRG(object):

    def _finalize_coef(self):
        self.coef_ *= self.coef_scale_
        self.coef_scale_.fill(1.0)

    def _fit(self, X, Y, sample_weight):
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)

        n_samples, n_features = X.shape
        rng = self._get_random_state()
        loss = self._get_loss(sample_weight)
        n_vectors = Y.shape[1]
        n_inner = int(self.n_inner * n_samples)
        ds = get_dataset(X, order="c")

        self.coef_ = np.zeros((n_vectors, n_features), dtype=np.float64)
        full_grad = np.zeros_like(self.coef_)
        grad = np.zeros((n_vectors, n_samples), dtype=np.float64)
        self.coef_scale_ = np.ones(n_vectors, dtype=np.float64)

        for i in xrange(n_vectors):
            y = Y[:, i]

            _svrg_fit(self, ds, y, self.coef_[i], self.coef_scale_[i:],
                      full_grad[i], grad[i], self.eta, self.alpha, loss,
                      self.max_iter, n_inner, self.tol, self.verbose,
                      self.callback, rng)

        return self


class SVRGClassifier(BaseClassifier, _BaseSVRG):
    """
    Estimator for learning linear classifiers by SVRG.

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

    def _get_loss(self, sample_weight):
        losses = {
            "modified_huber": ModifiedHuber(),
            "smooth_hinge": SmoothHinge(self.gamma),
            "squared_hinge": SquaredHinge(1.0),
            "log": Log(),
            "squared": SquaredLoss(),
        }
        loss = losses[self.loss]
        if sample_weight is not None:
            loss = WeightedLoss(sample_weight, loss)
        return loss

    def fit(self, X, y, sample_weight=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples,)
            Target vector relative to X.

        sample_weight : array-like, shape (n_samples,) optional
            Array of weights that are assigned to individual samples.
            If not provided, then each sample is given unit weight.

        Returns
        -------
        self : object
            Returns self.
        """
        self.label_binarizer_ = LabelBinarizer(neg_label=-1, pos_label=1)
        Y = np.asfortranarray(self.label_binarizer_.fit_transform(y),
                              dtype=np.float64)
        return self._fit(X, Y, sample_weight)


class SVRGRegressor(BaseRegressor, _BaseSVRG):
    """
    Estimator for learning linear regressors by SVRG.

    Solves the following objective:

        minimize_w  1 / n_samples * \sum_i loss(w^T x_i, y_i)
                    + alpha * 0.5 * ||w||^2_2
    """

    def __init__(self, eta=1.0, alpha=1.0, loss="squared", gamma=1.0,
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

    def _get_loss(self, sample_weight):
        losses = {
            "squared": SquaredLoss(),
        }
        loss = losses[self.loss]
        if sample_weight is not None:
            loss = WeightedLoss(sample_weight, loss)
        return loss

    def fit(self, X, y, sample_weight=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples,)
            Target vector relative to X.

        sample_weight : array-like, shape (n_samples,) optional
            Array of weights that are assigned to individual samples.
            If not provided, then each sample is given unit weight.

        Returns
        -------
        self : object
            Returns self.
        """
        self.outputs_2d_ = len(y.shape) > 1
        Y = y.reshape(-1, 1) if not self.outputs_2d_ else y
        Y = Y.astype(np.float64)
        return self._fit(X, Y, sample_weight)
