# Author: Mathieu Blondel
# License: BSD

import numpy as np
from scipy.linalg import svd
from lightning.impl.prox_fast import prox_tv1d


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
    def projection(self, coef, alpha, L):
        tmp = np.zeros_like(coef)
        for i in range(tmp.shape[0]):
            tmp[i, :] = prox_tv1d(coef[i, :], alpha / L)  # operates inplace
        return tmp

    def regularization(self, coef):
        return np.sum(np.abs(np.diff(coef)))
