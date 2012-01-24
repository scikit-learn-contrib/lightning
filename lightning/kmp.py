# Author: Mathieu Blondel
# License: BSD

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_random_state
from sklearn.externals.joblib import Parallel, delayed

from .primal import _dictionary


class SquaredLoss(object):

    def negative_gradient(self, y_true, y_pred):
        return y_true - y_pred

    def line_search(self, y, y_pred, column):
        squared_norm = np.sum(column ** 2)
        residuals = y - y_pred
        return np.dot(column, residuals) / squared_norm


def _fit_one(estimator, loss, K, y, n_nonzero_coefs, norms,
             refit, n_refit, check_duplicates):
    n_samples = K.shape[0]
    dictionary_size = K.shape[1]
    coef = np.zeros(dictionary_size, dtype=np.float64)
    selected = np.zeros(dictionary_size, dtype=bool)
    y_pred = np.zeros(n_samples, dtype=np.float64)

    if loss is None:
        residuals = y.copy()

    for i in range(n_nonzero_coefs):
        # compute pseudo-residuals if needed
        if loss is not None:
            residuals = loss.negative_gradient(y, y_pred)

        # select best basis
        dots = np.dot(K.T, residuals)
        dots /= norms
        if check_duplicates:
            dots[selected] = -np.inf
        best = np.argmax(dots)
        selected[best] = True

        if refit == "backfitting" and i % n_refit == 0 and n_refit != 0:
            # fit the selected coefficient and the previous ones too
            K_subset = K[:, selected]
            estimator.fit(K_subset, y)
            coef[selected] = estimator.coef_.ravel()

            if loss is None:
                residuals = y - estimator.predict(K_subset)
            else:
                y_pred = estimator.predict(K_subset)
        else:
            # find coefficient for the selected basis only
            if loss is None:
                alpha =  dots[best] / norms[best]
            else:
                alpha = loss.line_search(y, y_pred, K[:, best])

            before = coef[best]
            coef[best] += alpha
            diff = coef[best] - before

            if loss is None:
                residuals -= diff * K[:, best]
            else:
                y_pred += diff * K[:, best]

    return coef


class KernelMatchingPursuit(BaseEstimator, ClassifierMixin):

    def __init__(self,
                 n_nonzero_coefs,
                 loss=None,
                 # dictionary
                 dictionary_size=None,
                 check_duplicates=False,
                 # back-fitting
                 refit=None,
                 n_refit=1,
                 estimator=None,
                 # metric
                 metric="linear", gamma=0.1, coef0=1, degree=4,
                 # misc
                 random_state=None, verbose=0, n_jobs=1):
        if n_nonzero_coefs < 0:
            raise AttributeError("n_nonzero_coefs should be > 0.")

        self.n_nonzero_coefs = n_nonzero_coefs
        self.loss = loss
        self.dictionary_size = dictionary_size
        self.check_duplicates = check_duplicates
        self.refit = refit
        self.n_refit = n_refit
        self.estimator = estimator
        self.metric = metric
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
        self.random_state = random_state
        self.verbose = verbose
        self.n_jobs = n_jobs

    def _kernel_params(self):
        return {"gamma" : self.gamma,
                "degree" : self.degree,
                "coef0" : self.coef0}

    def _get_estimator(self):
        if self.estimator is None:
            estimator = LinearRegression()
        else:
            estimator = clone(self.estimator)
        return estimator

    def _get_loss(self):
        if self.loss == "squared":
            return SquaredLoss()
        else:
            return None

    def fit(self, X, y):
        random_state = check_random_state(self.random_state)

        self.lb_ = LabelBinarizer()
        Y = self.lb_.fit_transform(y)

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

        n = self.lb_.classes_.shape[0]
        n = 1 if n == 2 else n
        coef = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                delayed(_fit_one)(self._get_estimator(), self._get_loss(),
                                 K, Y[:, i], n_nonzero_coefs, norms,
                                 self.refit, self.n_refit,
                                 self.check_duplicates)
                for i in xrange(n))

        self.coef_ = np.array(coef)

        # trim unused basis functions
        used_basis = np.sum(self.coef_ != 0, axis=0, dtype=bool)
        self.coef_ = self.coef_[:, used_basis]
        self.dictionary_ = dictionary[used_basis]

        return self

    def decision_function(self, X):
        K = pairwise_kernels(X, self.dictionary_, metric=self.metric,
                             filter_params=True, **self._kernel_params())
        return np.dot(K, self.coef_.T)

    def predict(self, X):
        pred = self.decision_function(X)
        return self.lb_.inverse_transform(pred, threshold=0.5)

