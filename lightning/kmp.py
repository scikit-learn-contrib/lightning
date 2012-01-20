# Author: Mathieu Blondel
# License: BSD

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_random_state

from .primal import _dictionary

class KernelMatchingPursuit(BaseEstimator, ClassifierMixin):

    def __init__(self,
                 n_nonzero_coefs,
                 # dictionary
                 dictionary_size=None,
                 # back-fitting
                 refit=None,
                 alpha = 0,
                 # metric
                 metric="linear", gamma=0.1, coef0=1, degree=4,
                 # misc
                 random_state=None, verbose=0):
        if n_nonzero_coefs < 0:
            raise AttributeError("n_nonzero_coefs should be > 0.")

        self.n_nonzero_coefs = n_nonzero_coefs
        self.dictionary_size = dictionary_size
        self.refit = refit
        self.alpha = alpha
        self.metric = metric
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
        self.random_state = random_state
        self.verbose = verbose


    def _kernel_params(self):
        return {"gamma" : self.gamma,
                "degree" : self.degree,
                "coef0" : self.coef0}

    def fit(self, X, y):
        random_state = check_random_state(self.random_state)

        self.lb_ = LabelBinarizer()
        Y = self.lb_.fit_transform(y)
        y = Y[:, 0]

        n_nonzero_coefs = self.n_nonzero_coefs
        if 0 < n_nonzero_coefs and n_nonzero_coefs <= 1:
            n_nonzero_coefs = int(n_nonzero_coefs * X.shape[0])

        if self.verbose: print "Creating dictionary..."
        dictionary = _dictionary(X, self.dictionary_size, random_state)

        if n_nonzero_coefs > dictionary.shape[0]:
            raise AttributeError("n_nonzero_coefs cannot be bigger than "
                                 "dictionary_size.")

        if self.verbose: print "Computing kernel..."
        K = pairwise_kernels(X, dictionary, metric=self.metric,
                             filter_params=True, **self._kernel_params())

        # FIXME: this allocates a lot of intermediary memory
        norms = np.sqrt(np.sum(K ** 2, axis=0))

        coef = np.zeros(n_nonzero_coefs, dtype=np.float64)
        indices = np.zeros(n_nonzero_coefs, dtype=np.int)
        residuals = y.copy()

        lm = LinearRegression() if self.alpha == 0 else Ridge(alpha=self.alpha)

        for i in range(n_nonzero_coefs):
            dots = np.dot(K.T, residuals)
            dots /= norms
            indices[i] = np.argmax(dots)

            if self.refit is None:
                coef[i]= dots[indices[i]] / norms[indices[i]]
                residuals -= coef[i] * K[:, indices[i]]
            elif self.refit == "backfitting":
                K_subset = K[:, indices[:i+1]]
                lm.fit(K_subset, y)
                coef[:i+1] = lm.coef_.ravel()
                residuals = y - lm.predict(K_subset)

        self.coef_ = coef
        self.dictionary_ = dictionary[indices]

        return self

    def predict(self, X):
        K = pairwise_kernels(X, self.dictionary_, metric=self.metric,
                             filter_params=True, **self._kernel_params())
        pred = np.dot(K, self.coef_)
        return self.lb_.inverse_transform(pred, threshold=0.5)

