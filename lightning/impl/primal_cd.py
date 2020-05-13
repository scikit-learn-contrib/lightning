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

from joblib import Parallel, delayed
from six.moves import xrange

from .base import BaseClassifier
from .base import BaseRegressor

from .dataset_fast import get_dataset
from .primal_cd_fast import _primal_cd

from .primal_cd_fast import Squared
from .primal_cd_fast import SmoothHinge
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
            "smooth_hinge": SmoothHinge(**params),
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
        }
        return penalties[self.penalty]

    def _init_errors(self, Y):
        n_samples, n_vectors = Y.shape
        if self.loss == "squared":
            self.errors_ = -Y.T
        else:
            self.errors_ = np.ones((n_vectors, n_samples), dtype=np.float64)


class CDClassifier(_BaseCD, BaseClassifier):
    """Estimator for learning linear classifiers by (block) coordinate descent.

    The objective functions considered take the form

    minimize F(W) = C * L(W) + alpha * R(W),

    where L(W) is a loss term and R(W) is a penalty term.

    Parameters
    ----------
    loss : str, 'squared_hinge', 'log', 'modified_huber', 'squared'
        The loss function to be used.

    penalty : str, 'l2', 'l1', 'l1/l2'
        The penalty to be used.

        - l2: ridge
        - l1: lasso
        - l1/l2: group lasso

    multiclass : bool
        Whether to use a direct multiclass formulation (True) or one-vs-rest
        (False). Direct formulations are only available for loss='squared_hinge'
        and loss='log'.

    C : float
        Weight of the loss term.

    alpha : float
        Weight of the penalty term.

    max_iter : int
        Maximum number of iterations to perform.

    tol : float
        Tolerance of the stopping criterion.

    termination : str, 'violation_sum', 'violation_max'
        Stopping criterion to use.

    shrinking : bool
        Whether to activate shrinking or not.

    max_steps : int or "auto"
        Maximum number of steps to use during the line search. Use max_steps=0
        to use a constant step size instead of the line search. Use
        max_steps="auto" to let CDClassifier choose the best value.

    sigma : float
        Constant used in the line search sufficient decrease condition.

    beta : float
        Multiplicative constant used in the backtracking line search.

    warm_start : bool
        Whether to activate warm-start or not.

    debiasing : bool
        Whether to refit the model using l2 penalty (only useful if penalty='l1'
        or penalty='l1/l2').

    Cd : float
        Value of `C` when doing debiasing.

    warm_debiasing : bool
        Whether to warm-start the model or not when doing debiasing.

    selection : str, 'cyclic', 'uniform'
        Strategy to use for selecting coordinates.

    permute : bool
        Whether to permute coordinates or not before cycling (only when
        selection='cyclic').

    callback : callable
        Callback function.

    n_calls : int
        Frequency with which `callback` must be called.

    random_state : RandomState or int
        The seed of the pseudo random number generator to use.

    verbose : int
        Verbosity level.

    n_jobs : int
        Number of CPU's to be used when `multiclass=False` and when
        penalty is a non group-lasso penalty. By default use one CPU.
        If set to -1, use all CPU's

    Example
    -------

    The following example demonstrates how to learn a classification
    model with a multiclass squared hinge loss and an l1/l2 penalty.

    >>> from sklearn.datasets import fetch_20newsgroups_vectorized
    >>> from lightning.classification import CDClassifier
    >>> bunch = fetch_20newsgroups_vectorized(subset="all")
    >>> X, y = bunch.data, bunch.target
    >>> clf = CDClassifier(penalty="l1/l2",
                           loss="squared_hinge",
                           multiclass=True,
                           max_iter=20,
                           alpha=1e-4,
                           C=1.0 / X.shape[0],
                           tol=1e-3,
                           random_state=0).fit(X, y)
    >>> accuracy = clf.score(X, y)

    References
    ----------
    Block Coordinate Descent Algorithms for Large-scale Sparse Multiclass
    Classification.  Mathieu Blondel, Kazuhiro Seki, and Kuniaki Uehara.
    Machine Learning, May 2013.
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
                 random_state=None, verbose=0, n_jobs=1):
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
        self.n_jobs = n_jobs

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
        rs = self._get_random_state()

        # Create dataset
        ds = get_dataset(X, order="fortran")
        n_samples = ds.get_n_samples()
        n_features = ds.get_n_features()

        if self.penalty != "l1/l2" and self.multiclass:
            raise NotImplementedError("True multiclass options not implemented "
                                      "for non group-lasso(l1/l2) penalties.")

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
            model = _primal_cd(self, self.coef_, self.errors_,
                              ds, y, Y, -1, self.multiclass,
                              indices, 12, self._get_loss(),
                              self.selection, self.permute, self.termination,
                              self.C, self.alpha,
                              self.max_iter, max_steps,
                              self.shrinking, vinit,
                              rs, tol, self.callback, self.n_calls,
                              self.verbose)
            viol = model[0]
            if self.warm_start and len(self.violation_init_) == 0:
                self.violation_init_[0] = viol

        elif self.penalty in ("l1", "l2", "nn"):
            penalty = self._get_penalty()

            n_pos = np.zeros(n_vectors)
            vinit = self.C / self.C_init * np.ones_like(n_pos)
            for k in xrange(n_vectors):
                n_pos[k] = np.sum(Y[:, k] == 1)
                vinit[k] *= self.violation_init_.get(k, 0)
            n_neg = n_samples - n_pos
            tol = self.tol * np.maximum(np.minimum(n_pos, n_neg), 1) / n_samples

            jobs = (delayed(_primal_cd)(self, self.coef_, self.errors_,
                                        ds, y, Y, k, False,
                                        indices, penalty, self._get_loss(),
                                        self.selection, self.permute,
                                        self.termination,
                                        self.C, self.alpha,
                                        self.max_iter, max_steps,
                                        self.shrinking, vinit[k],
                                        rs, tol[k], self.callback, self.n_calls,
                                        self.verbose)
                    for k in xrange(n_vectors))
            model = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(jobs)
            viol, coefs, errors = zip(*model)
            self.coef_ = np.asarray(coefs)
            self.errors_ = np.asarray(errors)

            for k in range(n_vectors):
                if self.warm_start and not k in self.violation_init_:
                    self.violation_init_[k] = viol[k]

        if self.debiasing:
            nz = self.coef_ != 0

            if not self.warm_debiasing:
                self.coef_ = np.zeros((n_vectors, n_features), dtype=np.float64)
                self._init_errors(Y)

            indices = np.arange(n_features, dtype=np.int32)
            jobs = (delayed(_primal_cd)(
                           self, self.coef_, self.errors_,
                           ds, y, Y, k, False,
                           indices[nz[k]], 2, self._get_loss(),
                           "cyclic", self.permute,
                           "violation_sum",
                           self.Cd, 1.0,
                           self.max_iter, max_steps,
                           False, 0,
                           rs, self.tol, self.callback, self.n_calls,
                           self.verbose
                            )
                    for k in xrange(n_vectors))
            model = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(jobs)
            viol, coefs, errors = zip(*model)
            self.coef_ = np.asarray(coefs)
            self.errors_ = np.asarray(errors)

        return self


class CDRegressor(_BaseCD, BaseRegressor):
    """Estimator for learning linear regressors by (block) coordinate descent.

    The objective functions considered take the form

    minimize F(W) = C * L(W) + alpha * R(W),

    where L(W) is a loss term and R(W) is a penalty term.

    Parameters
    ----------
    loss : str, 'squared'
        The loss function to be used.

    penalty : str, 'l2', 'l1', 'l1/l2', 'nnl1', 'nnl2'
        The penalty to be used.

        - l2: ridge
        - l1: lasso
        - l1/l2: group lasso
        - nnl1: non-negative constraints + l1 penalty
        - nnl2: non-negative constraints + l2 penalty

    For other parameters, see `CDClassifier`.
    """

    def __init__(self, C=1.0, alpha=1.0,
                 loss="squared", penalty="l2",
                 max_iter=50, tol=1e-3, termination="violation_sum",
                 shrinking=True,
                 max_steps=30, sigma=0.01, beta=0.5,
                 warm_start=False, debiasing=False, Cd=1.0,
                 warm_debiasing=False,
                 selection="cyclic", permute=True,
                 callback=None, n_calls=100,
                 random_state=None, verbose=0, n_jobs=1):
        self.C = C
        self.alpha = alpha
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
        self.n_jobs = n_jobs

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
        if not self.warm_start or self.coef_ is None:
            self.C_init = self.C
            self.coef_ = np.zeros((n_vectors, n_features), dtype=np.float64)
            self._init_errors(Y)

        self.intercept_ = np.zeros(n_vectors, dtype=np.float64)
        indices = np.arange(n_features, dtype=np.int32)


        if self.penalty == "l1/l2":
            vinit = self.violation_init_.get(0, 0) * self.C / self.C_init
            model = _primal_cd(self, self.coef_, self.errors_,
                              ds, y, Y, -1, False,
                              indices, 12, self._get_loss(),
                              self.selection, self.permute, self.termination,
                              self.C, self.alpha,
                              self.max_iter, self.max_steps,
                              self.shrinking, vinit,
                              rs, self.tol, self.callback, self.n_calls,
                              self.verbose)
            viol = model[0]
            if self.warm_start and len(self.violation_init_) == 0:
                self.violation_init_[0] = viol
        else:
            penalty = self._get_penalty()
            vinit = np.asarray([self.violation_init_.get(k, 0)
                    for k in xrange(n_vectors)]) * self.C / self.C_init

            jobs = (delayed(_primal_cd)(self, self.coef_, self.errors_,
                                       ds, y, Y, k, False,
                                       indices, penalty, self._get_loss(),
                                       self.selection, self.permute,
                                       self.termination,
                                       self.C, self.alpha,
                                       self.max_iter, self.max_steps,
                                       self.shrinking, vinit[k],
                                       rs, self.tol, self.callback, self.n_calls,
                                       self.verbose)
                    for k in xrange(n_vectors))

            model = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(jobs)
            viol, self.coef_, self.error_ = zip(*model)
            self.coef_ = np.asarray(self.coef_)
            self.error_ = np.asarray(self.error_)

            if self.warm_start and not n_vectors in self.violation_init_:
                self.violation_init_[n_vectors] = viol

        return self
