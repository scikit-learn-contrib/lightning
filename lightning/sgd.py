# Author: Mathieu Blondel
# License: BSD

import warnings

import numpy as np

from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin
from sklearn.utils import check_random_state
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.validation import assert_all_finite

from .base import BaseClassifier
from .base import BaseRegressor

from .dataset_fast import get_dataset

from .sgd_fast import _binary_sgd
from .sgd_fast import _multiclass_sgd

from .sgd_fast import ModifiedHuber
from .sgd_fast import Hinge
from .sgd_fast import SquaredHinge
from .sgd_fast import Log
from .sgd_fast import SparseLog
from .sgd_fast import SquaredLoss
from .sgd_fast import Huber
from .sgd_fast import EpsilonInsensitive

from .sgd_fast import MulticlassLog
from .sgd_fast import MulticlassHinge
from .sgd_fast import MulticlassSquaredHinge


class BaseSGD(object):

    def _get_penalty(self):
        penalties = {
            "nn": -1,
            "nnl1": -1,
            "nnl2": -2,
            "l1": 1,
            "l2": 2,
            "l1/l2": 12
        }
        return penalties[self.penalty]

    def _get_learning_rate(self):
        learning_rates = {"constant": 1, "pegasos": 2, "invscaling": 3}
        return learning_rates[self.learning_rate]


class SGDClassifier(BaseClassifier, ClassifierMixin, BaseSGD):

    def __init__(self, loss="hinge", penalty="l2",
                 multiclass=False, alpha=0.01,
                 learning_rate="pegasos", eta0=0.03, power_t=0.5,
                 epsilon=0.01, fit_intercept=True, intercept_decay=1.0,
                 max_iter=10, shuffle=True, random_state=None,
                 callback=None, n_calls=100,
                 cache_mb=500, verbose=0, n_jobs=1):
        self.loss = loss
        self.penalty = penalty
        self.multiclass = multiclass
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.power_t = power_t
        self.epsilon = epsilon
        self.fit_intercept = fit_intercept
        self.intercept_decay = intercept_decay
        self.max_iter = max_iter
        self.shuffle = shuffle
        self.random_state = random_state
        self.callback = callback
        self.n_calls = n_calls
        self.cache_mb = cache_mb
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.coef_ = None

    def _get_loss(self):
        if self.multiclass:
            losses = {
                "log": MulticlassLog(),
                "hinge": MulticlassHinge(),
                "squared_hinge": MulticlassSquaredHinge(),
            }
        else:
            losses = {
                "modified_huber": ModifiedHuber(),
                "hinge": Hinge(1.0),
                "squared_hinge": SquaredHinge(1.0),
                "perceptron": Hinge(0.0),
                "log": Log(),
                "sparse_log": SparseLog(),
                "squared": SquaredLoss(),
                "huber": Huber(self.epsilon),
                "epsilon_insensitive": EpsilonInsensitive(self.epsilon)
            }
        return losses[self.loss]

    def fit(self, X, y):
        rs = check_random_state(self.random_state)

        reencode = self.multiclass
        y, n_classes, n_vectors = self._set_label_transformers(y, reencode)

        ds = get_dataset(X)
        n_samples = ds.get_n_samples()
        n_features = ds.get_n_features()
        self.coef_ = np.zeros((n_vectors, n_features), dtype=np.float64)

        self.intercept_ = np.zeros(n_vectors, dtype=np.float64)

        loss = self._get_loss()
        penalty = self._get_penalty()

        if n_vectors == 1 or not self.multiclass:
            Y = np.asfortranarray(self.label_binarizer_.fit_transform(y),
                                  dtype=np.float64)
            for i in xrange(n_vectors):
                _binary_sgd(self,
                            self.coef_, self.intercept_, i,
                            ds, Y[:, i], loss, penalty,
                            self.alpha,
                            self._get_learning_rate(),
                            self.eta0, self.power_t,
                            self.fit_intercept,
                            self.intercept_decay,
                            int(self.max_iter * n_samples), self.shuffle, rs,
                            self.callback, self.n_calls, self.verbose)

        elif self.multiclass:
            _multiclass_sgd(self, self.coef_, self.intercept_,
                            ds, y.astype(np.int32), loss, penalty,
                            self.alpha, self._get_learning_rate(),
                            self.eta0, self.power_t, self.fit_intercept,
                            self.intercept_decay,
                            int(self.max_iter * n_samples),
                            self.shuffle, rs, self.callback, self.n_calls,
                            self.verbose)

        else:
            raise ValueError("Wrong value for multiclass.")

        try:
            assert_all_finite(self.coef_)
        except ValueError:
            warnings.warn("coef_ contains infinite values")

        return self


class SGDRegressor(BaseRegressor, RegressorMixin, BaseSGD):

    def __init__(self, loss="squared", penalty="l2",
                 alpha=0.01,
                 learning_rate="pegasos", eta0=0.03, power_t=0.5,
                 epsilon=0.01, fit_intercept=True, intercept_decay=1.0,
                 max_iter=10, shuffle=True, random_state=None,
                 callback=None, n_calls=100,
                 cache_mb=500, verbose=0, n_jobs=1):
        self.loss = loss
        self.penalty = penalty
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.power_t = power_t
        self.epsilon = epsilon
        self.fit_intercept = fit_intercept
        self.intercept_decay = intercept_decay
        self.max_iter = max_iter
        self.shuffle = shuffle
        self.random_state = random_state
        self.callback = callback
        self.n_calls = n_calls
        self.cache_mb = cache_mb
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.coef_ = None

    def _get_loss(self):
        losses = {
            "squared": SquaredLoss(),
            "huber": Huber(self.epsilon),
            "epsilon_insensitive": EpsilonInsensitive(self.epsilon)
        }
        return losses[self.loss]

    def fit(self, X, y):
        rs = check_random_state(self.random_state)

        ds = get_dataset(X)
        n_samples = ds.get_n_samples()
        n_features = ds.get_n_features()

        self.outputs_2d_ = len(y.shape) == 2
        if self.outputs_2d_:
            Y = y
        else:
            Y = y.reshape(-1, 1)
        Y = np.asfortranarray(Y)
        n_vectors = Y.shape[1]
        self.coef_ = np.zeros((n_vectors, n_features), dtype=np.float64)
        self.intercept_ = np.zeros(n_vectors, dtype=np.float64)

        loss = self._get_loss()
        penalty = self._get_penalty()

        for k in xrange(n_vectors):
            _binary_sgd(self,
                        self.coef_, self.intercept_, k,
                        ds, Y[:, k], loss, penalty, self.alpha,
                        self._get_learning_rate(),
                        self.eta0, self.power_t,
                        self.fit_intercept,
                        self.intercept_decay,
                        int(self.max_iter * n_samples), self.shuffle, rs,
                        self.callback, self.n_calls, self.verbose)

        try:
            assert_all_finite(self.coef_)
        except ValueError:
            warnings.warn("coef_ contains infinite values")

        return self

    def predict(self, X):
        try:
            assert_all_finite(self.coef_)
            pred = safe_sparse_dot(X, self.coef_.T)
        except ValueError:
            n_samples = X.shape[0]
            n_vectors = self.coef_.shape[0]
            pred = np.zeros((n_samples, n_vectors))

        if not self.outputs_2d_:
            pred = pred.ravel()

        return pred
