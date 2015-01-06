# Author: Mathieu Blondel
# License: BSD

import numpy as np

from sklearn.utils import check_random_state
from sklearn.preprocessing import LabelBinarizer

from .base import BaseClassifier, BaseRegressor
from .dataset_fast import get_dataset
from .adagrad_fast import _adagrad_fit

from .sgd_fast import ModifiedHuber
from .sgd_fast import Hinge
from .sgd_fast import SmoothHinge
from .sgd_fast import SquaredHinge
from .sgd_fast import Log
from .sgd_fast import SquaredLoss
from .sgd_fast import EpsilonInsensitive
from .sgd_fast import Huber


class _BaseAdagrad(object):

    def _fit(self, X, Y):
        n_samples, n_features = X.shape
        rng = check_random_state(self.random_state)
        n_vectors = Y.shape[1]
        ds = get_dataset(X, order="c")

        self.coef_ = np.zeros((n_vectors, n_features), dtype=np.float64)
        self.g_sum_ = np.zeros((n_vectors, n_features), dtype=np.float64)
        self.g_norms_ = np.zeros((n_vectors, n_features), dtype=np.float64)

        delta = 0
        alpha1 = self.l1_ratio * self.alpha
        alpha2 = (1 - self.l1_ratio) * self.alpha
        loss = self._get_loss()
        n_calls = n_samples if self.n_calls is None else self.n_calls

        for i in xrange(n_vectors):
            _adagrad_fit(self, ds, Y[:, i], self.coef_[i], self.g_sum_[i],
                         self.g_norms_[i], loss, self.eta, delta, alpha1,
                         alpha2, self.n_iter, self.shuffle, self.callback,
                         n_calls, rng)

        return self


class AdaGradClassifier(BaseClassifier, _BaseAdagrad):
    """
    Estimator for learning linear classifiers by AdaGrad.

    Solves the following objective:

        minimize_w  1 / n_samples * \sum_i loss(w^T x_i, y_i)
                    + alpha * l1_ratio * ||w||_1
                    + alpha * (1 - l1_ratio) * 0.5 * ||w||^2_2
    """

    def __init__(self, eta=1.0, alpha=1.0, l1_ratio=0, loss="hinge", gamma=1.0,
                 n_iter=10, shuffle=True, callback=None, n_calls=None,
                 random_state=None):
        self.eta = eta
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.loss = loss
        self.gamma = gamma
        self.n_iter = n_iter
        self.shuffle = shuffle
        self.callback = callback
        self.n_calls = n_calls
        self.random_state = random_state

    def _get_loss(self):
        losses = {
            "modified_huber": ModifiedHuber(),
            "hinge": Hinge(1.0),
            "smooth_hinge": SmoothHinge(self.gamma),
            "squared_hinge": SquaredHinge(1.0),
            "perceptron": Hinge(0.0),
            "log": Log(),
            "squared": SquaredLoss(),
        }
        return losses[self.loss]

    def fit(self, X, y):
        self.label_binarizer_ = LabelBinarizer(neg_label=-1, pos_label=1)
        Y = np.asfortranarray(self.label_binarizer_.fit_transform(y),
                              dtype=np.float64)
        return self._fit(X, Y)


class AdaGradRegressor(BaseRegressor, _BaseAdagrad):
    """
    Estimator for learning linear regressors by AdaGrad.

    Solves the following objective:

        minimize_w  1 / n_samples * \sum_i loss(w^T x_i, y_i)
                    + alpha * l1_ratio * ||w||_1
                    + alpha * (1 - l1_ratio) * 0.5 * ||w||^2_2
    """

    def __init__(self, eta=1.0, alpha=1.0, l1_ratio=0, loss="squared",
                 gamma=1.0, epsilon=0, n_iter=10, shuffle=True, callback=None,
                 n_calls=None, random_state=None):
        self.eta = eta
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.loss = loss
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_iter = n_iter
        self.shuffle = shuffle
        self.callback = callback
        self.n_calls = n_calls
        self.random_state = random_state

    def _get_loss(self):
        losses = {
            "squared": SquaredLoss(),
            "huber": Huber(self.epsilon),
            "epsilon_insensitive": EpsilonInsensitive(self.epsilon),
            "absolute": EpsilonInsensitive(0)
        }
        return losses[self.loss]

    def fit(self, X, y):
        self.outputs_2d_ = len(y.shape) > 1
        Y = y.reshape(-1, 1) if not self.outputs_2d_ else y
        Y = Y.astype(np.float64)
        return self._fit(X, Y)
