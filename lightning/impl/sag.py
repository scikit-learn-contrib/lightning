# Author: Mathieu Blondel
#         Arnaud Rachez
#         Fabian Pedregosa
# License: BSD

import numpy as np

from sklearn.utils.extmath import row_norms
from sklearn.externals.six.moves import xrange

from .base import BaseClassifier, BaseRegressor
from .dataset_fast import get_dataset
from .sag_fast import _sag_fit

from .sgd_fast import ModifiedHuber
from .sgd_fast import SmoothHinge
from .sgd_fast import SquaredHinge
from .sgd_fast import Log
from .sgd_fast import SquaredLoss
from .sag_fast import L1Penalty


def get_auto_step_size(X, alpha, loss, gamma=None, sample_weight=None):
    """Compute automatic step size for SAG solver
    Stepsize computed using the following objective:
        minimize_w  1 / n_samples * \sum_i loss(w^T x_i, y_i)
                    + alpha * 0.5 * ||w||^2_2
    Parameters
    ----------
    X : ndarray
        Array of samples x_i.
    alpha : float
        Constant that multiplies the l2 penalty term.
    loss : string, in {"log", "squared"}
        The loss function used in SAG solver.

    Returns
    -------
    step_size : float
        Step size used in SAG/SAGA solver.
    """
    if sample_weight is None:
        weighted_norms = row_norms(X, squared=True)
    else:
        weighted_norms = sample_weight * row_norms(X, squared=True)
    L = np.max(weighted_norms)
    n_samples = X.shape[0]

    if loss == 'log':
        # inverse Lipschitz constant for log loss
        lipschitz_constant = 0.25 * L + alpha
    elif loss == 'squared':
        lipschitz_constant = L + alpha
    elif loss == 'modified_huber':
        lipschitz_constant = 2 * L + alpha
    elif loss == 'smooth_hinge':
        lipschitz_constant = L + gamma + alpha
    elif loss == 'squared_hinge':
        lipschitz_constant = 2 * L + alpha
    else:
        raise ValueError("`auto` stepsize is only available for `squared` or "
                         "`log` losses (got `%s` loss). Please specify a "
                         "stepsize." % loss)
    return 1.0 / lipschitz_constant


class _BaseSAG(object):

    def _get_loss(self):
        losses = {
            "modified_huber": ModifiedHuber(),
            "smooth_hinge": SmoothHinge(self.gamma),
            "squared_hinge": SquaredHinge(1.0),
            "log": Log(),
            "squared": SquaredLoss(),
        }
        return losses[self.loss]

    def _get_penalty(self):
        if isinstance(self.penalty, str):
            # l2 penalty is governed by the alpha keyword in `_sag_fit`.
            # beta governs the strength of the penalties below.
            penalties = {
                "l1": L1Penalty(),
            }
            return penalties[self.penalty]
        else:
            return self.penalty

    def _finalize_coef(self):
        self.coef_ *= self.coef_scale_
        self.coef_scale_.fill(1.0)

    def _fit(self, X, Y, sample_weight):
        n_samples, n_features = X.shape
        rng = self._get_random_state()
        adaptive_step_size = False

        if sample_weight is None:
            sample_weight = np.ones(n_samples, dtype=np.float64)
        else:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)

        if self.eta is None or self.eta in ('auto', 'line-search'):
            step_size = get_auto_step_size(
                    X, self.alpha, self.loss, self.gamma, sample_weight=sample_weight)
            if self.verbose > 0:
                print("Auto stepsize: %s" % self.eta)
            if self.eta == 'line-search':
                self.eta = step_size
                adaptive_step_size = True
            else:
                self.eta = step_size

        loss = self._get_loss()
        penalty = self._get_penalty()
        n_vectors = Y.shape[1]
        n_inner = int(self.n_inner * n_samples)
        ds = get_dataset(X, order="c")

        self.coef_ = np.zeros((n_vectors, n_features), dtype=np.float64)
        self.coef_scale_ = np.ones(n_vectors, dtype=np.float64)
        grad = np.zeros((n_vectors, n_samples), dtype=np.float64)

        for i in xrange(n_vectors):
            y = Y[:, i]

            _sag_fit(self, ds, y, self.coef_[i], self.coef_scale_[i:], grad[i],
                     sample_weight, self.eta, self.alpha, self.beta, loss, penalty,
                     self.max_iter, n_inner, self.tol, self.verbose,
                     self.callback, rng, self.is_saga, adaptive_step_size)

        return self


class SAGClassifier(BaseClassifier, _BaseSAG):
    """
    Estimator for learning linear classifiers by SAG.

    Solves the following objective:

        minimize_w  1 / n_samples * \sum_i loss(w^T x_i, y_i)
                    + alpha * 0.5 * ||w||^2_2

    Parameters
    ----------
    eta : float or {'auto', 'line-search'}, defaults to 'auto'
        step size for the gradient updates. If set to 'auto',
        this will calculate a step size based on the input data.
        If set to 'line-search', it will perform a line-search
        to find the step size based for the current iteration.
    alpha : float
        amount of squared L2 regularization
    beta : float
        amount of regularization for the penalty term
    loss : string
        loss to use in the objective function. Can be one of
        "smooth_hinge", "squared_hinge" or "log" (for logistic loss).
    gamma : float
        gamma parameter in the "smooth_hinge" loss (not used for other
        loss functions)
    max_iter : int
        maximum number of outer iterations (also known as epochs).
    tol : float
        stopping criterion tolerance.
    verbose : int
        verbosity level. Set positive to print progress information.
    callback : callable or None
        if given, callback(self) will be called on each outer iteration
        (epoch).
    random_state: int or RandomState
        Pseudo-random number generator state used for random sampling.
    """

    def __init__(self, eta='auto', alpha=1.0, beta=0.0, loss="smooth_hinge",
                 penalty=None, gamma=1.0, max_iter=10, n_inner=1.0, tol=1e-3,
                 verbose=0, callback=None, random_state=None):
        self.eta = eta
        self.alpha = alpha
        self.beta = beta
        self.loss = loss
        self.penalty = penalty
        self.gamma = gamma
        self.max_iter = max_iter
        self.n_inner = n_inner
        self.tol = tol
        self.verbose = verbose
        self.callback = callback
        self.random_state = random_state
        self.is_saga = False

    def fit(self, X, y, sample_weight=None):
        if not self.is_saga and self.penalty is not None:
            raise ValueError('Penalties in SAGClassifier. Please use '
                             'SAGAClassifier instead.'
                             '.')
        self._set_label_transformers(y, neg_label=-1)[0]
        y_binary = self.label_binarizer_.transform(y).astype(np.float64)
        return self._fit(X, y_binary, sample_weight)


class SAGAClassifier(SAGClassifier):
    """
    Estimator for learning linear classifiers by SAGA.

    Solves the following objective:

        minimize_w  1 / n_samples * \sum_i loss(w^T x_i, y_i)
                    + alpha * 0.5 * ||w||^2_2 + beta * penalty(w)

    Parameters
    ----------
    eta : float or {'auto', 'line-search'}, defaults to 'auto'
        step size for the gradient updates. If set to 'auto',
        this will calculate a step size based on the input data.
        If set to 'line-search', it will perform a line-search
        to find the step size based for the current iteration.
    alpha : float
        amount of squared L2 regularization
    beta : float
        amount of regularization for the penalty term
    loss : string
        loss to use in the objective function. Can be one of
        "smooth_hinge", "squared_hinge" or "log" (for logistic loss).
    penalty : string or Penalty object
        penalty term to use in the objective function. Can be "l1"
        or a custom Penalty object (object defined in
        lightning/impl/sag_fast.pxd)
    gamma : float
        gamma parameter in the "smooth_hinge" loss (not used for other
        loss functions)
    max_iter : int
        maximum number of outer iterations (also known as epochs).
    tol : float
        stopping criterion tolerance.
    verbose : int
        verbosity level. Set positive to print progress information.
    callback : callable or None
        if given, callback(self) will be called on each outer iteration
        (epoch).
    random_state: int or RandomState
        Pseudo-random number generator state used for random sampling.
    """

    def __init__(self, eta='auto', alpha=1.0, beta=0.0, loss="smooth_hinge",
                 penalty=None, gamma=1.0,  max_iter=10, n_inner=1.0,
                 tol=1e-3, verbose=0, callback=None, random_state=None):
            super(SAGAClassifier, self).__init__(
                eta=eta, alpha=alpha, beta=beta, loss=loss, penalty=penalty,
                gamma=gamma, max_iter=max_iter, n_inner=n_inner, tol=tol,
                verbose=verbose, callback=callback, random_state=random_state)
            self.is_saga = True


class SAGRegressor(BaseRegressor, _BaseSAG):
    """
    Estimator for learning linear regressors by SAG.

    Solves the following objective:

        minimize_w  1 / n_samples * \sum_i loss(w^T x_i, y_i)
                    + alpha * 0.5 * ||w||^2_2

    Parameters
    ----------
    eta : float or {'auto', 'line-search'}, defaults to 'auto'
        step size for the gradient updates. If set to 'auto',
        this will calculate a step size based on the input data.
        If set to 'line-search', it will perform a line-search
        to find the step size based for the current iteration.
    alpha : float
        amount of squared L2 regularization.
    beta : float
        amount of regularization for the penalty term.
    loss : string
        loss to use in the objective function. Can be "modified_huber" or
        "squared".
    max_iter : int
        maximum number of outer iterations (also known as epochs).
    tol : float
        stopping criterion tolerance.
    verbose : int
        verbosity level. Set positive to print progress information.
    callback : callable or None
        if given, callback(self) will be called on each outer iteration
        (epoch).
    random_state: int or RandomState
        Pseudo-random number generator state used for random sampling.
    """

    def __init__(self, eta='auto', alpha=1.0, beta=0.0, loss="smooth_hinge",
                 penalty=None, gamma=1.0, max_iter=10, n_inner=1.0, tol=1e-3,
                 verbose=0, callback=None, random_state=None):
        self.eta = eta
        self.alpha = alpha
        self.beta = beta
        self.loss = loss
        self.penalty = penalty
        self.gamma = gamma
        self.max_iter = max_iter
        self.n_inner = n_inner
        self.tol = tol
        self.verbose = verbose
        self.callback = callback
        self.random_state = random_state
        self.is_saga = False

    def fit(self, X, y, sample_weight=None):
        if not self.is_saga and self.penalty is not None:
            raise ValueError('Penalties are not supported in SAGRegressor. '
                             'Please use SAGARegressor instead.')
        self.outputs_2d_ = len(y.shape) > 1
        Y = y.reshape(-1, 1) if not self.outputs_2d_ else y
        Y = Y.astype(np.float64)
        return self._fit(X, Y, sample_weight=sample_weight)


class SAGARegressor(SAGRegressor):
    """
    Estimator for learning linear regressors by SAG.

    Solves the following objective:

        minimize_w  1 / n_samples * \sum_i loss(w^T x_i, y_i)
                    + alpha * 0.5 * ||w||^2_2 + beta * penalty(w)

    Parameters
    ----------
    eta : float or {'auto', 'line-search'}, defaults to 'auto'
        step size for the gradient updates. If set to 'auto',
        this will calculate a step size based on the input data.
        If set to 'line-search', it will perform a line-search
        to find the step size based for the current iteration.
    alpha : float
        amount of squared L2 regularization
    beta : float
        amount of regularization for the penalty term
    loss : string
        loss to use in the objective function. Can be "modified_huber" or
        "squared".
    penalty : string or Penalty object
        penalty term to use in the objective function. Can be "l1"
        or a custom Penalty object (object defined in
        lightning/impl/sag_fast.pxd)
    max_iter : int
        maximum number of outer iterations (also known as epochs).
    tol : float
        stopping criterion tolerance.
    verbose : int
        verbosity level. Set positive to print progress information.
    callback : callable or None
        if given, callback(self) will be called on each outer iteration
        (epoch).
    random_state: int or RandomState
        Pseudo-random number generator state used for random sampling.
    """

    def __init__(self, eta='auto', alpha=1.0, beta=0.0, loss="smooth_hinge",
                 penalty="l1", max_iter=10, n_inner=1.0, tol=1e-3,
                 verbose=0, callback=None, random_state=None):
            super(SAGARegressor, self).__init__(
                eta=eta, alpha=alpha, beta=beta, loss=loss, penalty=penalty,
                gamma=1.0, max_iter=max_iter, n_inner=n_inner, tol=tol,
                verbose=verbose, callback=callback, random_state=random_state)
            self.is_saga = True
