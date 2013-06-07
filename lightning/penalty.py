# Author: Mathieu Blondel
# License: BSD

import numpy as np
from scipy.linalg import svd


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
