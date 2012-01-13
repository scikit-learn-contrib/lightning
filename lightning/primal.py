# Author: Mathieu Blondel
# License: BSD

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils import check_random_state, safe_mask

def _dictionary(X, dictionary_size, random_state):
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

def _trim_dictionary(estimator, dictionary):
    if not hasattr(estimator, "coef_"):
        raise AttributeError("Base estimator doesn't have a coef_ attribute.")
    used_basis = np.sum(estimator.coef_ != 0, axis=0, dtype=bool)
    used_basis = safe_mask(dictionary, used_basis)
    estimator.coef_ = estimator.coef_[:, used_basis]
    return dictionary[used_basis]

def fit_primal(estimator, metric, dictionary_size, X, y, random_state):
    dictionary = _dictionary(X, dictionary_size, random_state)
    estimator = clone(estimator)
    K = pairwise_kernels(X, dictionary, metric=metric)
    estimator.fit(K, y)
    return dictionary, estimator

def predict_primal(estimator, dictionary, metric, X):
    K = pairwise_kernels(X, dictionary, metric=metric)
    return estimator.predict(K)

class PrimalClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, estimator, metric="linear", dictionary_size=None,
                 trim_dictionary=True,
                 random_state=None):
        self.estimator_ = estimator
        self.metric = metric
        self.dictionary_size = dictionary_size
        self.trim_dictionary = trim_dictionary
        self.random_state = random_state
        self.estimators_ = []

    def fit(self, X, y):
        random_state = check_random_state(self.random_state)

        self.dictionary_, self.estimator_ = \
                fit_primal(self.estimator_,
                           self.metric,
                           self.dictionary_size,
                           X, y,
                           self.random_state)

        if self.trim_dictionary:
            self.dictionary_ = \
                    _trim_dictionary(self.estimator_, self.dictionary_)

        return self

    def predict(self, X):
        return predict_primal(self.estimator_,
                              self.dictionary_,
                              self.metric,
                              X)

