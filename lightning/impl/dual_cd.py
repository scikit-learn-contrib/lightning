"""
==========================================
Dual Coordinate Descent Solvers
==========================================

This module provides coordinate descent solvers for support vector machines
(SVMs) and support vector regression (SVR) with L2 regularization.
"""

# Author: Mathieu Blondel
# License: BSD

import numpy as np

from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import add_dummy_feature
from six.moves import xrange

from .base import BaseClassifier, BaseRegressor
from .dataset_fast import get_dataset
from .dual_cd_fast import _dual_cd
from .dual_cd_fast import _dual_cd_auc
from .dual_cd_fast import _dual_cd_svr


class LinearSVC(BaseClassifier):
    """Estimator for learning linear support vector machine by coordinate
    descent in the dual.

    Parameters
    ----------
    loss : str, 'hinge', 'squared_hinge'
        The loss function to be used.

    criterion : str, 'accuracy', 'auc'
        Whether to optimize for classification accuracy or AUC.

    C : float
        Weight of the loss term.

    max_iter : int
        Maximum number of iterations to perform.

    tol : float
        Tolerance of the stopping criterion.

    shrinking : bool
        Whether to activate shrinking or not.

    warm_start : bool
        Whether to activate warm-start or not.

    permute : bool
        Whether to permute coordinates or not before cycling.

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

    The following example demonstrates how to learn a classification
    model:

    >>> from sklearn.datasets import fetch_20newsgroups_vectorized
    >>> from lightning.classification import LinearSVC
    >>> bunch = fetch_20newsgroups_vectorized(subset="all")
    >>> X, y = bunch.data, bunch.target
    >>> clf = LinearSVC().fit(X, y)
    >>> accuracy = clf.score(X, y)
    """

    def __init__(self, C=1.0, loss="hinge", criterion="accuracy",
                 max_iter=1000, tol=1e-3,
                 permute=True, shrinking=True, warm_start=False,
                 random_state=None, callback=None, n_calls=100, verbose=0):
        self.C = C
        self.loss = loss
        self.criterion = criterion
        self.max_iter = max_iter
        self.tol = tol
        self.permute = permute
        self.shrinking = shrinking
        self.warm_start = warm_start
        self.random_state = random_state
        self.callback = callback
        self.n_calls = n_calls
        self.verbose = verbose
        self.coef_ = None

    def _get_loss(self):
        loss = {"l1": 1,
                "hinge": 1,
                "l2": 2,
                "squared_hinge": 2}
        return loss[self.loss]

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
        n_samples, n_features = X.shape
        rs = self._get_random_state()

        self._set_label_transformers(y)
        Y = np.asfortranarray(self.label_binarizer_.transform(y),
                              dtype=np.float64)
        n_vectors = Y.shape[1]

        ds = get_dataset(X)

        if not self.warm_start or self.coef_ is None:
            self.coef_ = np.zeros((n_vectors, n_features), dtype=np.float64)
            if self.criterion == "accuracy":
                self.dual_coef_ = np.zeros((n_vectors, n_samples),
                                           dtype=np.float64)

        for i in xrange(n_vectors):
            if self.criterion == "accuracy":
                _dual_cd(self, self.coef_[i], self.dual_coef_[i],
                         ds, Y[:, i], self.permute,
                         self.C, self._get_loss(), self.max_iter, rs, self.tol,
                         self.shrinking, self.callback, self.n_calls,
                         verbose=self.verbose)
            else:
                _dual_cd_auc(self, self.coef_[i], ds, Y[:, i],
                             self.C, self._get_loss(), self.max_iter, rs,
                             self.verbose)

        return self


class LinearSVR(BaseRegressor):
    """Estimator for learning a linear support vector regressor by coordinate
    descent in the dual.

    Parameters
    ----------
    loss : str, 'epsilon_insensitive', 'squared_epsilon_insensitive'
        The loss function to be used.

    C : float
        Weight of the loss term.

    epsilon : float
        Parameter of the epsilon-insensitive loss.

    max_iter : int
        Maximum number of iterations to perform.

    tol : float
        Tolerance of the stopping criterion.

    fit_intercept : bool
        Whether to fit an intercept term or not.

    warm_start : bool
        Whether to activate warm-start or not.

    permute : bool
        Whether to permute coordinates or not before cycling.

    callback : callable
        Callback function.

    n_calls : int
        Frequency with which `callback` must be called.

    random_state : RandomState or int
        The seed of the pseudo random number generator to use.

    verbose : int
        Verbosity level.
    """

    def __init__(self, C=1.0, epsilon=0, loss="epsilon_insensitive",
                 max_iter=1000, tol=1e-3, fit_intercept=False,
                 permute=True, warm_start=False,
                 random_state=None, callback=None, n_calls=100, verbose=0):
        self.C = C
        self.epsilon = epsilon
        self.loss = loss
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.permute = permute
        self.warm_start = warm_start
        self.random_state = random_state
        self.callback = callback
        self.n_calls = n_calls
        self.verbose = verbose
        self.coef_ = None

    def _get_loss(self):
        loss = {"l1": 1,
                "epsilon_insensitive": 1,
                "l2": 2,
                "squared_epsilon_insensitive": 2}
        return loss[self.loss]

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
        self : regressor
            Returns self.
        """
        if self.fit_intercept:
            X = add_dummy_feature(X)

        n_samples, n_features = X.shape
        rs = self._get_random_state()

        self.outputs_2d_ = len(y.shape) == 2
        if self.outputs_2d_:
            Y = y
        else:
            Y = y.reshape(-1, 1)
        Y = np.asfortranarray(Y)
        n_vectors = Y.shape[1]

        ds = get_dataset(X)

        if not self.warm_start or self.coef_ is None:
            self.coef_ = np.zeros((n_vectors, n_features), dtype=np.float64)
            self.dual_coef_ = np.zeros((n_vectors, n_samples),
                                       dtype=np.float64)

        for i in xrange(n_vectors):
            _dual_cd_svr(self, self.coef_[i], self.dual_coef_[i],
                         ds, Y[:, i], self.permute,
                         self.C, self.epsilon, self._get_loss(),
                         self.max_iter, rs, self.tol,
                         self.callback, self.n_calls,
                         verbose=self.verbose)

        if self.fit_intercept:
            self.intercept_ = self.coef_[:, 0]
            self.coef_ = self.coef_[:, 1:]

        return self
