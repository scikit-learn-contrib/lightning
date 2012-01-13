# Author: Mathieu Blondel
# License: BSD

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils import check_random_state

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

def fit_primal(estimator, metric, dictionary_size, X, y, random_state):
    dictionary = _dictionary(X, dictionary_size, random_state)
    estimator = clone(estimator)
    K = pairwise_kernels(X, dictionary, metric=metric)
    estimator.fit(K, y)
    return dictionary, [estimator]

def predict_primal(estimator, dictionary, metric, X):
    K = pairwise_kernels(X, dictionary, metric=metric)
    return estimator.predict(K)

class PrimalClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, estimator, metric="linear", dictionary_size=None,
                 random_state=None):
        self.estimator = estimator
        self.metric = metric
        self.dictionary_size = dictionary_size
        self.random_state = random_state
        self.estimators_ = []

    def fit(self, X, y):
        random_state = check_random_state(self.random_state)
        self.dictionary_, self.estimators_ = \
                fit_primal(self.estimator,
                           self.metric,
                           self.dictionary_size,
                           X, y,
                           self.random_state)
        return self

    def predict(self, X):
        return predict_primal(self.estimators_[0],
                              self.dictionary_,
                              self.metric,
                              X)

