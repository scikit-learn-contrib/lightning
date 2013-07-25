"""
==========================================
Primal (Block) Coordinate Descent Solvers
==========================================

This module provides (block) coordinate descent solvers for a variety of loss
functions and penalties.
"""

# Author: Mathieu Blondel
# License: BSD

import numpy as np

from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin

from .base import BaseClassifier
from .base import BaseRegressor

from .dataset_fast import get_dataset
from .primal_cd_fast import _primal_cd

from .primal_cd_fast import Squared
from .primal_cd_fast import SquaredHinge
from .primal_cd_fast import ModifiedHuber
from .primal_cd_fast import Log


class _BaseCD(object):

    def _get_loss(self):
        params = {"max_steps": self._get_max_steps(),
                  "sigma": self.sigma,
                  "beta": self.beta,
                  "verbose": self.verbose}
        losses = {
            "squared": Squared(verbose=self.verbose),
            "squared_hinge": SquaredHinge(**params),
            "modified_huber": ModifiedHuber(**params),
            "log": Log(**params),
        }

        return losses[self.loss]

    def _get_max_steps(self):
        if self.max_steps == "auto":
            if self.loss == "log":
                max_steps = 0
            else:
                max_steps = 30
        else:
            max_steps = self.max_steps
        return max_steps

    def _get_penalty(self):
        penalties = {
            "l1": 1,
            "l2": 2,
            "nn": -1,
            "nnl1": -1,
            "nnl2": -2
        }
        return penalties[self.penalty]

    def _init_errors(self, Y):
        n_samples, n_vectors = Y.shape
        if self.loss == "squared":
            self.errors_ = -Y.T
        else:
            self.errors_ = np.ones((n_vectors, n_samples), dtype=np.float64)


class CDClassifier(_BaseCD, BaseClassifier, ClassifierMixin):
    """Estimator for learning linear classifiers by coordinate descent (CD).

    The objective functions considered take the form

    minimize F(W) = C * L(W) + alpha * R(W),

    where L(W) is a loss term and R(W) is a penalty term.

    Parameters
    ----------
    loss : str, 'squared_hinge', 'log', 'modified_huber', 'squared'
        The loss function to be used.

    penalty: str, 'l2', 'l1', 'l1/l2'
        The penalty to be used.
        - l2: ridge
        - l1: lasso
        - l1/l2: group lasso

    multiclass: bool
        Whether to use a direct multiclass formulation (True) or one-vs-rest
        (False). Direct formulations are only available for loss='squared_hinge'
        and loss='log'.

    C: float
        Weight of the loss term.

    alpha: float
        Weight of the penalty term.

    max_iter: int
        Maximum number of iterations to perform.

    tol: float
        Tolerance of the stopping criterion.

    termination: str, 'violation_sum', 'violation_max'
        Stopping criterion to use.

    shrinking: bool
        Whether to activate shrinking or not.

    max_steps: int or "auto"
        Maximum number of steps to use during the line search. Use max_steps=0
        to use a constant step size instead of the line search. Use
        max_steps="auto" to let CDClassifier choose the best value.

    sigma: float
        Constant used in the line search sufficient decrease condition.

    beta: float
        Multiplicative constant used in the backtracking line search.

    warm_start: bool
        Whether to activate warm-start or not.

    debiasing: bool
        Whether to refit the model using l2 penalty (only useful if penalty='l1'
        or penalty='l1/l2').

    Cd: float
        Value of `C` when doing debiasing.

    warm_debiasing: bool
        Whether to warm-start the model or not when doing debiasing.

    selection: str, 'cyclic', 'uniform'
        Strategy to use for selecting coordinates.

    permute: bool
        Whether to permute coordinates or not before cycling (only when
        selection='cyclic').

    callback: callable
        Callback function.

    n_calls: int
        Frequency with which `callback` must be called.

    random_state: RandomState or int
        The seed of the pseudo random number generator to use.

    verbose: int
        Verbosity level.
    """

    def __init__(self, loss="squared_hinge", penalty="l2", multiclass=False,
                 C=1.0, alpha=1.0,
                 max_iter=50, tol=1e-3, termination="violation_sum",
                 shrinking=True,
                 max_steps="auto", sigma=0.01, beta=0.5,
                 warm_start=False, debiasing=False, Cd=1.0,
                 warm_debiasing=False,
                 selection="cyclic", permute=True,
                 callback=None, n_calls=100,
                 random_state=None, verbose=0):
        self.C = C
        self.alpha = alpha
        self.loss = loss
        self.penalty = penalty
        self.multiclass = multiclass
        self.max_iter = max_iter
        self.tol = tol
        self.termination = termination
        self.shrinking = shrinking
        self.max_steps = max_steps
        self.sigma = sigma
        self.beta = beta
        self.warm_start = warm_start
        self.debiasing = debiasing
        self.Cd = Cd
        self.warm_debiasing = warm_debiasing
        self.selection = selection
        self.permute = permute
        self.callback = callback
        self.n_calls = n_calls
        self.random_state = random_state
        self.verbose = verbose
        self.coef_ = None
        self.violation_init_ = {}

    def fit(self, X, y):
        rs = self._get_random_state()

        # Create dataset
        ds = get_dataset(X, order="fortran")
        n_samples = ds.get_n_samples()
        n_features = ds.get_n_features()

        # Create label transformers
        #neg_label = 0 if self.penalty == "nn" else -1
        reencode = self.penalty == "l1/l2"
        y, n_classes, n_vectors = self._set_label_transformers(y,
                                                               reencode,
                                                               neg_label=-1)
        Y = np.asfortranarray(self.label_binarizer_.transform(y),
                              dtype=np.float64)

        # Initialize coefficients
        if not self.warm_start or self.coef_ is None:
            self.C_init = self.C
            self.coef_ = np.zeros((n_vectors, n_features), dtype=np.float64)
            self._init_errors(Y)

        self.intercept_ = np.zeros(n_vectors, dtype=np.float64)
        indices = np.arange(n_features, dtype=np.int32)

        max_steps = self._get_max_steps()

        # Learning
        if self.penalty == "l1/l2":
            tol = self.tol
            #n_min = np.min(np.sum(Y == 1, axis=0))
            #tol *= max(n_min, 1) / n_samples

            vinit = self.violation_init_.get(0, 0) * self.C / self.C_init
            viol = _primal_cd(self, self.coef_, self.errors_,
                              ds, y, Y, -1, self.multiclass,
                              indices, 12, self._get_loss(),
                              self.selection, self.permute, self.termination,
                              self.C, self.alpha, 1e12,
                              self.max_iter, max_steps,
                              self.shrinking, vinit,
                              rs, tol, self.callback, self.n_calls,
                              self.verbose)
            if self.warm_start and len(self.violation_init_) == 0:
                self.violation_init_[0] = viol

        elif self.penalty in ("l1", "l2", "nn"):
            penalty = self._get_penalty()
            for k in xrange(n_vectors):
                n_pos = np.sum(Y[:, k] == 1)
                n_neg = n_samples - n_pos
                tol = self.tol * max(min(n_pos, n_neg), 1) / n_samples

                vinit = self.violation_init_.get(k, 0) * self.C / self.C_init
                viol = _primal_cd(self, self.coef_, self.errors_,
                                  ds, y, Y, k, False,
                                  indices, penalty, self._get_loss(),
                                  self.selection, self.permute,
                                  self.termination,
                                  self.C, self.alpha, 1e12,
                                  self.max_iter, max_steps,
                                  self.shrinking, vinit,
                                  rs, tol, self.callback, self.n_calls,
                                  self.verbose)

                if self.warm_start and not k in self.violation_init_:
                    self.violation_init_[k] = viol

        if self.debiasing:
            nz = self.coef_ != 0

            if not self.warm_debiasing:
                self.coef_ = np.zeros((n_vectors, n_features), dtype=np.float64)
                self._init_errors(Y)

            for k in xrange(n_vectors):
                indices = np.arange(n_features, dtype=np.int32)[nz[k]]
                _primal_cd(self, self.coef_, self.errors_,
                           ds, y, Y, k, False,
                           indices, 2, self._get_loss(),
                           "cyclic", self.permute,
                           "violation_sum",
                           self.Cd, 1.0, 1e12,
                           self.max_iter, max_steps,
                           False, 0,
                           rs, self.tol, self.callback, self.n_calls,
                           self.verbose)

        return self


class CDRegressor(_BaseCD, BaseRegressor, RegressorMixin):
    """Estimator for learning linear regressors by coordinate descent (CD).

    The objective functions considered take the form

    minimize F(W) = C * L(W) + alpha * R(W),

    where L(W) is a loss term and R(W) is a penalty term.

    Parameters
    ----------
    loss : str, 'squared'
        The loss function to be used.

    penalty: str, 'l2', 'l1', 'l1/l2', 'nnl1', 'nnl2'
        The penalty to be used.
        - l2: ridge
        - l1: lasso
        - l1/l2: group lasso
        - nnl1: non-negative constraints + l1 penalty
        - nnl2: non-negative constraints + l2 penalty

    U: float
        Upper bound on the weights (only when penalty='nnl1' or 'nnl2').


    For other parameters, see `CDClassifier`.
    """

    def __init__(self, C=1.0, alpha=1.0, U=1e12,
                 loss="squared", penalty="l2",
                 max_iter=50, tol=1e-3, termination="violation_sum",
                 shrinking=True,
                 max_steps=30, sigma=0.01, beta=0.5,
                 warm_start=False, debiasing=False, Cd=1.0,
                 warm_debiasing=False,
                 selection="cyclic", permute=True,
                 callback=None, n_calls=100,
                 random_state=None, verbose=0):
        self.C = C
        self.alpha = alpha
        self.U = U
        self.loss = loss
        self.penalty = penalty
        self.max_iter = max_iter
        self.tol = tol
        self.termination = termination
        self.shrinking = shrinking
        self.max_steps = max_steps
        self.sigma = sigma
        self.beta = beta
        self.warm_start = warm_start
        self.debiasing = debiasing
        self.Cd = Cd
        self.warm_debiasing = warm_debiasing
        self.selection = selection
        self.permute = permute
        self.callback = callback
        self.n_calls = n_calls
        self.random_state = random_state
        self.verbose = verbose
        self.coef_ = None
        self.violation_init_ = {}

    def fit(self, X, y):
        rs = self._get_random_state()

        # Create dataset
        ds = get_dataset(X, order="fortran")
        n_features = ds.get_n_features()

        self.outputs_2d_ = len(y.shape) == 2
        if self.outputs_2d_:
            Y = y
        else:
            Y = y.reshape(-1, 1)
        Y = np.asfortranarray(Y, dtype=np.float64)
        y = np.empty(0, dtype=np.int32)
        n_vectors = Y.shape[1]

        # Initialize coefficients
        if self.warm_start and self.coef_ is not None:
            if self.kernel:
                coef = np.zeros((n_vectors, n_features), dtype=np.float64)
                coef[:, self.support_indices_] = self.coef_
                self.coef_ = coef
        else:
            self.C_init = self.C
            self.coef_ = np.zeros((n_vectors, n_features), dtype=np.float64)
            self._init_errors(Y)

        self.intercept_ = np.zeros(n_vectors, dtype=np.float64)
        indices = np.arange(n_features, dtype=np.int32)

        penalty = self._get_penalty()
        for k in xrange(n_vectors):
            vinit = self.violation_init_.get(k, 0) * self.C / self.C_init
            viol = _primal_cd(self, self.coef_, self.errors_,
                              ds, y, Y, k, False,
                              indices, penalty, self._get_loss(),
                              self.selection, self.permute,
                              self.termination,
                              self.C, self.alpha, self.U,
                              self.max_iter, self.max_steps,
                              self.shrinking, vinit,
                              rs, self.tol, self.callback, self.n_calls,
                              self.verbose)

            if self.warm_start and not k in self.violation_init_:
                self.violation_init_[k] = viol

        return self
