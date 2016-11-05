# Author: Mathieu Blondel
# License: BSD

import numpy as np

from sklearn.utils.extmath import safe_sparse_dot
from sklearn.externals.six.moves import xrange

from .base import BaseClassifier, BaseRegressor

from .dataset_fast import get_dataset

from .loss_fast import Squared
from .loss_fast import SquaredHinge
from .loss_fast import MulticlassSquaredHinge
from .loss_fast import MulticlassLog

from .penalty import L1Penalty
from .penalty import L1L2Penalty
from .penalty import TracePenalty
from .penalty import SimplexConstraint
from .penalty import L1BallConstraint
from .penalty import TotalVariation1DPenalty
from .penalty import TotalVariation2DPenalty


class _BaseFista(object):

    def _get_penalty(self):
        if hasattr(self.penalty, 'projection'):
            return self.penalty
        penalties = {
            "l1": L1Penalty,
            "l1/l2": L1L2Penalty,
            "trace": TracePenalty,
            "simplex": SimplexConstraint,
            "l1-ball": L1BallConstraint,
            "tv1d": TotalVariation1DPenalty,
            "tv2d": TotalVariation2DPenalty
        }
        return penalties[self.penalty](*self.prox_args)

    def _get_objective(self, df, y, loss):
        return self.C * loss.objective(df, y)

    def _get_regularized_objective(self, df, y, loss, penalty, coef):
        obj = self._get_objective(df, y, loss)
        obj += self.alpha * penalty.regularization(coef)
        return obj

    def _get_quad_approx(self, coefa, coefb, objb, gradb, L, penalty):
        approx = objb
        diff = coefa - coefb
        approx += np.sum(diff * gradb)
        approx += L / 2 * np.sum(diff ** 2)
        approx += self.alpha * penalty.regularization(coefa)
        return approx

    def _fit(self, X, y, n_vectors):
        n_samples, n_features = X.shape
        loss = self._get_loss()
        penalty = self._get_penalty()
        ds = get_dataset(X)

        df = np.zeros((n_samples, n_vectors), dtype=np.float64)
        coef = np.zeros((n_vectors, n_features), dtype=np.float64)
        coefx = coef
        G = np.zeros((n_vectors, n_features), dtype=np.float64)

        obj = self._get_regularized_objective(df, y, loss, penalty, coef)

        if self.max_steps == 0:
            # No line search, need to use constant step size.
            L = self.C * loss.lipschitz_constant(ds, n_vectors)
        else:
            # Do not bother to compute the Lipschitz constant (expensive).
            L = 1.0

        t = 1.0
        for it in xrange(self.max_iter):
            old_obj = obj

            if self.verbose >= 1:
                print("Iter=%s, \tloss=%s" % (it+1, obj))

            # Save current values
            t_old = t
            coefx_old = coefx

            # Gradient
            G.fill(0.0)
            loss.gradient(df, ds, y, G)
            G *= self.C

            # Line search
            if self.max_steps > 0:
                objb = self._get_objective(df, y, loss)

            for tt in xrange(self.max_steps):
                # Solve
                coefx = coef - G / L
                coefx = penalty.projection(coefx, self.alpha, L)

                dfx = safe_sparse_dot(X, coefx.T)
                obj = self._get_regularized_objective(dfx, y, loss, penalty,
                                                      coefx)
                approx = self._get_quad_approx(coefx, coef, objb, G, L, penalty)

                accepted = obj <= approx

                # Sufficient decrease condition
                if accepted:
                    if self.verbose >= 2:
                        print("Accepted at", tt + 1)
                    break
                else:
                    L *= self.eta

            if self.max_steps == 0:
                coefx = coef - G / L
                coefx = penalty.projection(coefx, self.alpha, L)

            t = (1 + np.sqrt(1 + 4 * t_old * t_old) / 2)
            coef = coefx + (t_old - 1) / t * (coefx - coefx_old)
            df = safe_sparse_dot(X, coef.T)

            # Callback might need self.coef_.
            self.coef_ = coef
            if self.callback is not None:
                ret = self.callback(self)
                if ret is not None:
                    break

            # stop if change in last iteration is below tol
            if np.abs(old_obj - obj) < self.tol:
                if self.verbose >= 1:
                    print('Break, reached desired tolerance')
                break

        return self


class FistaClassifier(BaseClassifier, _BaseFista):
    """Estimator for learning linear classifiers by FISTA.

    The objective functions considered take the form

    minimize F(W) = C * L(W) + alpha * R(W),

    where L(W) is a loss term and R(W) is a penalty term.

    Parameters
    ----------
    loss : str, 'squared_hinge', 'log', 'modified_huber', 'squared'
        The loss function to be used.

    penalty : str or Penalty object, 'l2', 'l1', 'l1/l2', 'simplex'
        The penalty or constraint to be used.

        - l2: ridge
        - l1: lasso
        - l1/l2: group lasso
        - tv1d: 1-dimensional total variation (also known as fused lasso)
        - simplex: simplex constraint
        The method can also take an arbitrary Penalty object, i.e., an instance
        that implements methods projection regularization method (see file penalty.py)

    multiclass : bool
        Whether to use a direct multiclass formulation (True) or one-vs-rest
        (False).

    C : float
        Weight of the loss term.

    alpha : float
        Weight of the penalty term.

    max_iter : int
        Maximum number of iterations to perform.

    max_steps : int
        Maximum number of steps to use during the line search.

    sigma : float
        Constant used in the line search sufficient decrease condition.

    eta : float
         Decrease factor for line-search procedure. For example, eta=2.
         will decrease the step size by a factor of 2 at each iteration
         of the line-search routine.

    callback : callable
        Callback function.

    verbose : int
        Verbosity level.

    tol : float
        tolerance in the optimization scheme
    """

    def __init__(self, C=1.0, alpha=1.0, loss="squared_hinge", penalty="l1",
                 multiclass=False, max_iter=100, max_steps=30, eta=2.0,
                 sigma=1e-5, callback=None, verbose=0, prox_args=(), tol=1e-6):
        self.C = C
        self.alpha = alpha
        self.loss = loss
        self.penalty = penalty
        self.multiclass = multiclass
        self.max_iter = max_iter
        self.max_steps = max_steps
        self.eta = eta
        self.sigma = sigma
        self.callback = callback
        self.verbose = verbose
        self.prox_args = prox_args
        self.tol = tol

    def _get_loss(self):
        if self.multiclass:
            losses = {
                "squared_hinge": MulticlassSquaredHinge(),
                "log": MulticlassLog(),
                "log_margin": MulticlassLog(margin=1),
            }
        else:
            losses = {
                "squared_hinge": SquaredHinge(),
            }

        return losses[self.loss]

    def fit(self, X, y):
        y, _, n_vectors = self._set_label_transformers(y, reencode=True)

        if not self.multiclass:
            y = np.asfortranarray(self.label_binarizer_.transform(y),
                                  dtype=np.float64)
            n_vectors = y.shape[1]

        return self._fit(X, y, n_vectors)


class FistaRegressor(BaseRegressor, _BaseFista):
    """Estimator for learning linear classifiers by FISTA.

    The objective functions considered take the form

    minimize F(W) = C * L(W) + alpha * R(W),

    where L(W) is a loss term and R(W) is a penalty term.

    Parameters
    ----------
    penalty : str or Penalty object, {'l2', 'l1', 'l1/l2', 'simplex'}
        The penalty or constraint to be used.

        - l2: ridge
        - l1: lasso
        - l1/l2: group lasso
        - tv1d: 1-dimensional total variation (also known as fussed lasso)
        - simplex: simplex constraint
        The method can also take an arbitrary Penalty object, i.e., an instance
        that implements methods projection regularization method (see file penalty.py)


    C : float
        Weight of the loss term.

    alpha : float
        Weight of the penalty term.

    max_iter : int
        Maximum number of iterations to perform.

    max_steps : int
        Maximum number of steps to use during the line search.

    sigma : float
        Constant used in the line search sufficient decrease condition.

    eta : float
         Decrease factor for line-search procedure. For example, eta=2.
         will decrease the step size by a factor of 2 at each iteration
         of the line-search routine.

    callback : callable
        Callback function.

    verbose : int
        Verbosity level.

    tol : float
        Tolerance in the optimization scheme.
    """

    def __init__(self, C=1.0, alpha=1.0, penalty="l1", max_iter=100,
                 max_steps=30, eta=2.0, sigma=1e-5, callback=None, verbose=0,
                 prox_args=(), tol=1e-6):
        self.C = C
        self.alpha = alpha
        self.penalty = penalty
        self.max_iter = max_iter
        self.max_steps = max_steps
        self.eta = eta
        self.sigma = sigma
        self.callback = callback
        self.verbose = verbose
        self.prox_args = prox_args
        self.tol = tol

    def _get_loss(self):
        return Squared()

    def fit(self, X, y):
        self.outputs_2d_ = len(y.shape) > 1
        Y = y.reshape(-1, 1) if not self.outputs_2d_ else y
        Y = np.asfortranarray(Y.astype(np.float64))
        n_vectors = Y.shape[1]
        return self._fit(X, Y, n_vectors)
