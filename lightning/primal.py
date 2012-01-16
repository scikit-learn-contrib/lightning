# Author: Mathieu Blondel
# License: BSD

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils import check_random_state, safe_mask


def _dictionary(X, dictionary_size, random_state):
    # FIXME: support number of SVs proportional to n_samples in each class
    n_samples = X.shape[0]

    if not dictionary_size:
        return X

    if dictionary_size < 0:
        raise ValueError("dictionary_size must be a positive value.")

    if 0 < dictionary_size and dictionary_size < 1:
        dictionary_size = int(dictionary_size * n_samples)

    indices = np.arange(n_samples)
    random_state.shuffle(indices)

    return X[indices[:dictionary_size]]


def _trim_dictionary(estimator, dictionary, K=None):
    if not hasattr(estimator, "coef_"):
        raise AttributeError("Base estimator doesn't have a coef_ attribute.")
    used_basis = np.sum(estimator.coef_ != 0, axis=0, dtype=bool)
    used_basis = safe_mask(dictionary, used_basis)
    estimator.coef_ = estimator.coef_[:, used_basis]
    if K is not None:
        K = K[:, used_basis]
    return dictionary[used_basis], K


class PrimalClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, estimator, metric="linear", dictionary_size=None,
                 trim_dictionary=True, debiasing=False,
                 random_state=None, verbose=0,
                 gamma=0.1, coef0=1, degree=4):
        self.estimator_ = clone(estimator)
        self.metric = metric
        self.dictionary_size = dictionary_size
        self.trim_dictionary = trim_dictionary
        self.debiasing = debiasing
        self.random_state = random_state
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
        self.verbose = verbose

    def _params(self):
        return {"gamma" : self.gamma,
                "degree" : self.degree,
                "coef0" : self.coef0}

    def fit(self, X, y):
        random_state = check_random_state(self.random_state)

        if self.verbose: print "Creating dictionary..."
        self.dictionary_ = _dictionary(X, self.dictionary_size, random_state)

        if self.verbose: print "Computing kernel..."
        K = pairwise_kernels(X, self.dictionary_, metric=self.metric,
                             filter_params=True, **self._params())

        if self.verbose: print "Computing model..."
        self.estimator_.fit(K, y)

        if not self.debiasing:
            K = None

        if self.trim_dictionary:
            if self.verbose: print "Triming dictionary..."
            self.dictionary_, K = _trim_dictionary(self.estimator_,
                                                   self.dictionary_,
                                                   K)

        if self.debiasing:
            if self.verbose: print "Debiasing..."
            self.estimator_.set_params(**{"penalty": "l2"})
            self.estimator_.fit(K, y)

        return self

    def predict(self, X):
        K = pairwise_kernels(X, self.dictionary_, metric=self.metric,
                             filter_params=True, **self._params())
        return self.estimator_.predict(K)

    @property
    def coef_(self):
        return self.estimator_.coef_
