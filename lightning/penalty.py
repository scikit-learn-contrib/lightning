# Author: Mathieu Blondel
# License: BSD

import numpy as np

class L1Penalty(object):

    def projection(self, coef, alpha, L):
        return np.sign(coef) * np.maximum(np.abs(coef) - alpha / L, 0)

    def regularization(self, coef):
        return np.sum(np.abs(coef))


class L1L2Penalty(object):

    def projection(self, coef, alpha, L):
        n_features = coef.shape[1]
        l2norms = np.sqrt(np.sum(coef ** 2, axis=0))
        scales = np.maximum(1.0 - alpha / (L * l2norms), 0)
        coef *= scales
        return coef

    def regularization(self, coef):
        return np.sum(np.sqrt(np.sum(coef ** 2, axis=0)))


class NNConstraint(object):

    def projection(self, coef, alpha, L):
        return np.maximum(0, coef)

    def regularization(self, coef):
        return 0
