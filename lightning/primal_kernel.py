# Author: Mathieu Blondel
# License: BSD

import numpy as np

from sklearn.base import ClassifierMixin, clone
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_random_state
from sklearn.metrics.pairwise import pairwise_kernels

from .base import BaseLinearClassifier, BaseKernelClassifier

class HingeLoss(object):

    def objective(self, K, y, coef, C):
        value = 0.5 * np.dot(np.dot(K, coef), coef)
        losses = np.maximum(1 - y * np.dot(K, coef), 0) ** 2
        value += C * np.sum(losses)
        return value

    def derivative(self, K, y, coef, j, C):
        value = np.dot(coef, K[j])
        losses = np.maximum(1 - y * np.dot(K, coef), 0) ** 2
        value += -2 * C * np.sum(y * K[j] * losses)
        return value

    def second_derivative(self, K, y, coef, j, C):
        value = K[j, j]
        value += 2 * C * np.sum(K[j] ** 2)
        return value

class LogLoss(object):

    def objective(self, K, y, coef, C):
        value = 0.5 * np.dot(np.dot(K, coef), coef)
        losses = np.log(1 + np.exp(-y * np.dot(K, coef)))
        value += C * np.sum(losses)
        return value

    def derivative(self, K, y, coef, j, C):
        value = np.dot(coef, K[j])
        losses = y * np.dot(K, coef)
        losses = 1 / (1 + np.exp(-losses))
        value += C * np.sum(y * K[j] * (losses - 1))
        return value

    def second_derivative(self, K, y, coef, j, C):
        value = K[j, j]
        losses = y * np.dot(K, coef)
        losses = 1 / (1 + np.exp(-losses))
        value += C * np.sum(K[j] ** 2 * losses * (1- losses))
        return value


class ModifiedHuberLoss(object):

    def objective(self, K, y, coef, C):
        value = 0.5 * np.dot(np.dot(K, coef), coef)
        losses = y * np.dot(K, coef)
        cond = losses < -1
        not_cond = ~cond
        losses[cond] *= -4
        losses[not_cond] = np.maximum(1-losses[not_cond], 0) ** 2
        value += C * np.sum(losses)
        return value

    def derivative(self, K, y, coef, j, C):
        value = np.dot(coef, K[j])
        losses = y * np.dot(K, coef)
        cond = losses < -1
        not_cond = ~cond
        losses[cond] = -4
        losses[not_cond] = -2 * np.maximum(1 - losses[not_cond], 0)
        value += C * np.sum(y * K[j] * losses)
        return value

    def second_derivative(self, K, y, coef, j, C):
        value = K[j, j]
        losses = y * np.dot(K, coef)
        cond = np.logical_and(-1 <= losses, losses <= 1)
        value += 2 * C * np.sum(K[j, cond] ** 2)
        return value


class PrimalKernelSVC(BaseKernelClassifier, ClassifierMixin):

    def __init__(self, C=1.0, loss="hinge", max_iter=10, tol=1e-3,
                 kernel="linear", gamma=0.1, coef0=1, degree=4,
                 selection="permute", search_size=60,
                 termination="convergence", sv_upper_bound=1000,
                 cache_mb=500, warm_start=False, random_state=None,
                 callback=None, verbose=0, n_jobs=1):
        self.C = C
        self.loss = loss
        self.max_iter = max_iter
        self.tol = tol
        self.kernel = kernel
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
        self.selection = selection
        self.search_size = search_size
        self.termination = termination
        self.sv_upper_bound = sv_upper_bound
        self.cache_mb = cache_mb
        self.warm_start = warm_start
        self.random_state = random_state
        self.callback = callback
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.support_vectors_ = None
        self.coef_ = None

    def _get_loss(self):
        losses = {"hinge" : HingeLoss(),
                  "log" : LogLoss(),
                  "modified_huber" : ModifiedHuberLoss()}
        return losses[self.loss]

    def _solve_one(self, K, y, coef, j, loss):
        sigma = 0.01
        beta = 0.5
        L0 = loss.objective(K, y, coef, self.C)
        d = -loss.derivative(K, y, coef, j, self.C)
        d /= loss.second_derivative(K, y, coef, j, self.C)
        old_coef = coef[j]
        z = d

        for i in xrange(100):
            coef[j] = old_coef + z
            Li = loss.objective(K, y, coef, self.C)
            if Li - L0 <= -sigma * (z ** 2):
                break
            z *= beta

    def _fit_binary(self, K, y, coef, loss, rs):
        n_samples = K.shape[0]
        indices = np.arange(n_samples)
        rs.shuffle(indices)

        for t in xrange(self.max_iter * n_samples):
            j = indices[(t-1) % n_samples]
            self._solve_one(K, y, coef, j, loss)

        #good = y * np.dot(K, coef) > 1
        #indices = np.arange(n_samples)
        #indices = indices[~good]
        #coef[good] = 0
        #n_sv = len(indices)

        #for t in xrange(n_sv * 3):
            #j = indices[t % n_sv]
            #self._solve_one(K, y, coef, j)

    def fit(self, X, y):
        n_samples = X.shape[0]
        rs = check_random_state(self.random_state)

        self.label_binarizer_ = LabelBinarizer(neg_label=-1, pos_label=1)
        Y = self.label_binarizer_.fit_transform(y)
        self.classes_ = self.label_binarizer_.classes_.astype(np.int32)
        n_vectors = Y.shape[1]

        self.coef_ = np.zeros((n_vectors, n_samples), dtype=np.float64)
        #self.errors_ = np.ones((n_vectors, n_samples), dtype=np.float64)
        self.intercept_ = np.zeros(n_vectors, dtype=np.float64)

        K = pairwise_kernels(X, metric=self.kernel, filter_params=True,
                             **self._kernel_params())
        loss = self._get_loss()

        for i in xrange(n_vectors):
            self._fit_binary(K, Y[:, i], self.coef_[i], loss, rs)

        self._post_process(X)

        return self
