# Author: Mathieu Blondel
# License: BSD

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin, clone, is_classifier
from sklearn.cross_validation import check_cv
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

    def __init__(self,
                 # base_estimator
                 estimator,
                 # dictionary
                 dictionary_size=None, trim_dictionary=True,
                 # learning
                 debiasing=False,
                 # metric
                 metric="linear", gamma=0.1, coef0=1, degree=4,
                 # misc
                 random_state=None, verbose=0):
        self.estimator_ = estimator
        self.dictionary_size = dictionary_size
        self.trim_dictionary = trim_dictionary
        self.debiasing = debiasing
        self.metric = metric
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
        self.random_state = random_state
        self.verbose = verbose

    def _params(self):
        return {"gamma" : self.gamma,
                "degree" : self.degree,
                "coef0" : self.coef0}

    def _get_regularization_param(self):
        params = self.estimator_.get_params(deep=False)
        for param in ("C", "alpha"):
            if param in params:
                return (param, params[param])
        raise ValueError("Base estimator should have a parameter C or alpha.")

    def _fit_one(self, X, y, param):
        random_state = check_random_state(self.random_state)

        if self.verbose: print "Creating dictionary..."
        dictionary = _dictionary(X, self.dictionary_size, random_state)


        if self.verbose: print "Computing kernel..."
        K = pairwise_kernels(X, dictionary, metric=self.metric,
                             filter_params=True, **self._params())

        estimator = clone(self.estimator_)
        param_name, _ = self._get_regularization_param()
        estimator.set_params(**{param_name: param})
        estimator.fit(K, y)

        if self.trim_dictionary:
            if self.verbose: print "Triming dictionary..."
            dictionary, K = _trim_dictionary(estimator,
                                             dictionary,
                                             K)

        if self.debiasing:
            if self.verbose: print "Debiasing..."
            estimator.set_params(**{"penalty": "l2"})
            estimator.fit(K, y)

        self.dictionary_ = dictionary
        self.estimator_ = estimator

        return self

    def fit(self, X, y):
        param_name, param_value = self._get_regularization_param()
        return self._fit_one(X, y, param_value)

    def predict(self, X):
        K = pairwise_kernels(X, self.dictionary_, metric=self.metric,
                             filter_params=True, **self._params())
        return self.estimator_.predict(K)

    @property
    def coef_(self):
        return self.estimator_.coef_

    @property
    def n_support_(self):
        return np.sum(self.coef_ != 0, axis=1)


class PrimalClassifierCV(PrimalClassifier):

    def __init__(self,
                 # base_estimator
                 estimator,
                 # cv
                 cv=3,
                 params=None,
                 upper_bound=None,
                 # dictionary
                 dictionary_size=None, trim_dictionary=True,
                 # learning
                 debiasing=False,
                 # metric
                 metric="linear", gamma=0.1, coef0=1, degree=4,
                 # misc
                 random_state=None, verbose=0):

        if params is None:
            raise AttributeError("`params` must be a list of parameters.")

        self.estimator_ = estimator
        self.cv = cv
        self.params = params
        self.upper_bound = upper_bound
        self.dictionary_size = dictionary_size
        self.trim_dictionary = trim_dictionary
        self.debiasing = debiasing
        self.metric = metric
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
        self.random_state = random_state
        self.verbose = verbose

    def _fit_path(self, X_train, y_train, X_val, y_val):
        random_state = check_random_state(self.random_state)

        if self.verbose: print "Creating dictionary..."
        dictionary = _dictionary(X_train, self.dictionary_size, random_state)

        if self.verbose: print "Computing kernel..."
        K_train = pairwise_kernels(X_train, dictionary, metric=self.metric,
                                   filter_params=True, **self._params())

        K_val = pairwise_kernels(X_val, dictionary, metric=self.metric,
                                 filter_params=True, **self._params())

        scores = []
        estimators = []
        n_svs = []
        param_name, _ = self._get_regularization_param()
        estimator = clone(self.estimator_)

        for param in self.params:
            if self.verbose: print "Computing model for %s..." % str(param)
            estimator.set_params(**{param_name: param})
            estimator.fit(K_train, y_train)

            score = estimator.score(K_val, y_val)
            nsv = np.mean(np.sum(estimator.coef_ != 0, axis=0))
            # FIXME: we could stop here if nsv > upper_bound
            scores.append(score)
            estimators.append(estimator)
            n_svs.append(nsv)

        # Use the dictionary size if upper_bound is not provided
        upper_bound = self.upper_bound
        if upper_bound is None:
            upper_bound = dictionary.shape[0]

        scores = np.array(scores)
        n_svs = np.array(n_svs) / float(upper_bound)

        return estimators, scores, n_svs

    def fit(self, X, y):
        cv = check_cv(self.cv, X, y, is_classifier(self.estimator_))

        n_params = len(self.params)
        n_folds = 0
        scores = np.zeros(n_params, dtype=np.float64)
        n_svs = np.zeros(n_params, dtype=np.float64)
        for train, val in cv:
            ret = self._fit_path(X[train], y[train], X[val], y[val])
            scores += ret[1]
            n_svs += ret[2]
            n_folds += 1

        scores /= n_folds
        n_svs /= n_folds

        best_score = -np.inf
        best_param = None

        for i in range(n_params):
            if n_svs[i] > 1.0:
                break
            elif scores[i] > best_score:
                best_score = scores[i]
                best_param = self.params[i]


        if best_param is None:
            raise ValueError("Could not find a parameters that below "
                             "upper bound")

        self._fit_one(X, y, best_param)

        return self
