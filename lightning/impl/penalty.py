"""
In this file there are defined several proximal operators for common penalty
functions. These objects implement the following methods:

      * projection(self, coef, alpha, L)
          returns the value of the proximal operator for the penalty,
          where coef is a ndarray that contains the coefficients of the
          model, alpha is the amount of regularization and 1/L is the
          step size

      * regularization(self, coef)
          returns the value of the penalty at coef, where coef is a
          ndarray.
"""
# Author: Mathieu Blondel
#         Fabian Pedregosa
# License: BSD

import numpy as np
from scipy.linalg import svd
from lightning.impl.prox_fast import prox_tv1d, prox_tv2d


class L1Penalty(object):

    def projection(self, coef, alpha, L):
        return np.sign(coef) * np.maximum(np.abs(coef) - alpha / L, 0)

    def regularization(self, coef):
        return np.sum(np.abs(coef))


class L1L2Penalty(object):

    def projection(self, coef, alpha, L):
        l2norms = np.sqrt(np.sum(coef ** 2, axis=0))
        scales = np.maximum(1.0 - alpha / (L * l2norms), 0)
        coef *= scales
        return coef

    def regularization(self, coef):
        return np.sum(np.sqrt(np.sum(coef ** 2, axis=0)))


class TracePenalty(object):

    def projection(self, coef, alpha, L):
        U, s, V = svd(coef, full_matrices=False)
        s = np.maximum(s - alpha / L, 0)
        #return np.dot(np.dot(U, np.diag(s)), V)
        U *= s
        return np.dot(U, V)

    def regularization(self, coef):
        U, s, V = svd(coef, full_matrices=False)
        return np.sum(s)


class NNConstraint(object):

    def projection(self, coef, alpha, L):
        return np.maximum(0, coef)

    def regularization(self, coef):
        return 0


# See https://gist.github.com/mblondel/6f3b7aaad90606b98f71
# for more algorithms.
def project_simplex(v, z=1):
    if np.sum(v) <= z:
        return v

    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w


class SimplexConstraint(object):

    def projection(self, coef, alpha, L):
        return project_simplex(coef[0]).reshape(1,-1)

    def regularization(self, coef):
        return 0


def project_l1_ball(v, z=1):
    return np.sign(v) * project_simplex(np.abs(v), z)


class L1BallConstraint(object):

    def projection(self, coef, alpha, L):
        return project_l1_ball(coef[0], alpha).reshape(1,-1)

    def regularization(self, coef):
        return 0


class TotalVariation1DPenalty(object):
    """
    Proximal operator for the 1-D total variation penalty (also known
    as fussed lasso)
    """
    def projection(self, coef, alpha, L):
        tmp = np.empty_like(coef)
        for i in range(tmp.shape[0]):
            tmp[i] = prox_tv1d(coef[i], alpha / L)
        return tmp

    def regularization(self, coef):
        return np.sum(np.abs(np.diff(coef)))


class TotalVariation2DPenalty(object):
    """
    Proximal operator for the 2-D total variation penalty. This
    proximal operator is computed approximately using the
    Douglas-Rachford algorithm.

    Parameters
    ----------
    n_rows: int
        number of rows in the image

    n_cols: int
        number of columns in the image

    max_iter: int
        maximum number of iterations to compute this proximal
        operator.

    Misc
    -----
    Note that n_rows * n_cols needs to be equal to the size
    of the vector of coefficients.

    References
    ----------
    Barbero, Alvaro, and Suvrit Sra. "Modular proximal optimization for
    multidimensional total-variation regularization." arXiv preprint
    arXiv:1411.0589 (2014).
    """
    def __init__(self, n_rows, n_cols, max_iter=1000, tol=1e-12):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.max_iter = max_iter
        self.tol = tol

    def projection(self, coef, alpha, L):
        tmp = np.empty_like(coef)
        for i in range(tmp.shape[0]):
            tmp[i] = prox_tv2d(
                coef[i].reshape((self.n_rows, self.n_cols)),
                alpha / L, self.max_iter, self.tol).ravel()
        return tmp

    def regularization(self, coef):
        out = 0.0
        for i in range(coef.shape[0]):
            img = coef[i].reshape((self.n_rows, self.n_cols))
            tmp1 = np.abs(np.diff(img, axis=0))
            tmp2 = np.abs(np.diff(img, axis=1))
            out += tmp1.sum() + tmp2.sum()
        return out
