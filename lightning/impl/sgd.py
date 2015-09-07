"""
===================================
Stochastic Gradient Descent Solvers
===================================

This module provides SGD solvers for a variety of loss
functions and penalties.
"""
# Author: Mathieu Blondel
# License: BSD

import warnings

import numpy as np

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
from .sgd_fast import SquaredLoss
from .sgd_fast import Huber
from .sgd_fast import EpsilonInsensitive

from .sgd_fast import MulticlassLog
from .sgd_fast import MulticlassHinge
from .sgd_fast import MulticlassSquaredHinge

# fix for missing xrange in Python3
try:
    xrange
except NameError:
    xrange = range


class _BaseSGD(object):

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


class SGDClassifier(BaseClassifier, _BaseSGD):
    """Estimator for learning linear classifiers by SGD.

    Parameters
    ----------
    loss : str, 'hinge', 'squared_hinge', 'log', 'perceptron'
        Loss function to be used.

    penalty : str, 'l2', 'l1', 'l1/l2'
        The penalty to be used.

        - l2: ridge
        - l1: lasso
        - l1/l2: group lasso

    multiclass : bool
        Whether to use a direct multiclass formulation (True) or one-vs-rest
        (False). Direct formulations are only available for loss='hinge', 'squared_hinge'
        and 'log'.

    alpha : float
        Weight of the penalty term.

    learning_rate : 'pegasos', 'constant', 'invscaling'
        Learning schedule to use.

    eta0 : float
        Step size.

    power_t : float
        Power to be used (when learning_rate='invscaling').

    epsilon : float
        Value to be used for epsilon-insensitive loss.

    fit_intercept : bool
        Whether to fit the intercept or not.

    intercept_decay : float
        Value by which the intercept is multiplied (to regularize it).

    max_iter : int
        Maximum number of iterations to perform.

    shuffle : bool
        Whether to shuffle data.

    callback : callable
        Callback function.

    n_calls : int
        Frequency with which `callback` must be called.

    random_state : RandomState or int
        The seed of the pseudo random number generator to use.

    verbose : int
        Verbosity level.

    Example
    -------

    >>> from sklearn.datasets import fetch_20newsgroups_vectorized
    >>> from lightning.classification import SGDClassifier
    >>> bunch = fetch_20newsgroups_vectorized(subset="all")
    >>> X, y = bunch.data, bunch.target
    >>> clf = SGDClassifier().fit(X, y)
    >>> accuracy = clf.score(X, y)
    """

    def __init__(self, loss="hinge", penalty="l2",
                 multiclass=False, alpha=0.01,
                 learning_rate="pegasos", eta0=0.03, power_t=0.5,
                 epsilon=0.01, fit_intercept=True, intercept_decay=1.0,
                 max_iter=10, shuffle=True, random_state=None,
                 callback=None, n_calls=100, verbose=0):
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
        self.verbose = verbose
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
                "squared": SquaredLoss(),
                "huber": Huber(self.epsilon),
                "epsilon_insensitive": EpsilonInsensitive(self.epsilon)
            }
        return losses[self.loss]

    def fit(self, X, y):
        """Fit model according to X and y.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : classifier
            Returns self.
        """
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


class SGDRegressor(BaseRegressor, _BaseSGD):
    """Estimator for learning linear regressors by SGD.

    Parameters
    ----------
    loss : str, 'squared', 'epsilon_insensitive', 'huber'
        Loss function to be used.

    penalty : str, 'l2', 'l1', 'l1/l2'
        The penalty to be used.

        - l2: ridge
        - l1: lasso
        - l1/l2: group lasso

    alpha : float
        Weight of the penalty term.

    learning_rate : 'pegasos', 'constant', 'invscaling'
        Learning schedule to use.

    eta0 : float
        Step size.

    power_t : float
        Power to be used (when learning_rate='invscaling').

    epsilon : float
        Value to be used for epsilon-insensitive loss.

    fit_intercept : bool
        Whether to fit the intercept or not.

    intercept_decay : float
        Value by which the intercept is multiplied (to regularize it).

    max_iter : int
        Maximum number of iterations to perform.

    shuffle : bool
        Whether to shuffle data.

    callback : callable
        Callback function.

    n_calls : int
        Frequency with which `callback` must be called.

    random_state : RandomState or int
        The seed of the pseudo random number generator to use.

    verbose : int
        Verbosity level.
    """

    def __init__(self, loss="squared", penalty="l2",
                 alpha=0.01,
                 learning_rate="pegasos", eta0=0.03, power_t=0.5,
                 epsilon=0.01, fit_intercept=True, intercept_decay=1.0,
                 max_iter=10, shuffle=True, random_state=None,
                 callback=None, n_calls=100, verbose=0):
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
        self.verbose = verbose
        self.coef_ = None

    def _get_loss(self):
        losses = {
            "squared": SquaredLoss(),
            "huber": Huber(self.epsilon),
            "epsilon_insensitive": EpsilonInsensitive(self.epsilon)
        }
        return losses[self.loss]

    def fit(self, X, y):
        """Fit model according to X and y.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape = [n_samples] or [n_samples, n_targets]
            Target values.

        Returns
        -------
        self : regressor
            Returns self.
        """
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
        """
        Perform regression on an array of test vectors X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        p : array, shape = [n_samples]
            Predicted target values for X
        """
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
