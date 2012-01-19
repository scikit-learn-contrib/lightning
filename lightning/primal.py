# Author: Mathieu Blondel
# License: BSD

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin, clone, is_classifier
from sklearn.cross_validation import check_cv
from sklearn.grid_search import IterGrid
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


def _get_regularization_param(estimator):
    params = estimator.get_params(deep=True)
    for param in ("C", "alpha", "estimator__C", "estimator__alpha"):
        if param in params:
            return (param, params[param])
    raise ValueError("Estimator should have a parameter C or alpha.")


class BasePrimal(BaseEstimator):

    def _fit_one(self, X, y, regul_value, kernel_params):
        random_state = check_random_state(self.random_state)

        if self.verbose: print "Creating dictionary..."
        dictionary = _dictionary(X, self.dictionary_size, random_state)


        if self.verbose: print "Computing kernel..."
        K = pairwise_kernels(X, dictionary, metric=self.metric,
                             filter_params=True, **kernel_params)

        estimator = clone(self.estimator_)
        param_name, _ = _get_regularization_param(self.estimator_)
        estimator.set_params(**{param_name: regul_value})
        estimator.fit(K, y)

        if self.trim_dictionary:
            if self.verbose: print "Triming dictionary..."
            dictionary, K = _trim_dictionary(estimator,
                                             dictionary,
                                             K)

        if self.debiasing:
            if self.verbose: print "Debiasing..."
            estimator.set_params(penalty="l2")
            estimator.fit(K, y)

        self.dictionary_ = dictionary
        self.estimator_ = estimator

        return self

    @property
    def coef_(self):
        return self.estimator_.coef_

    @property
    def n_support_(self):
        return np.sum(self.coef_ != 0, axis=1)


class PrimalClassifier(BasePrimal, ClassifierMixin):

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

    def _kernel_params(self):
        return {"gamma" : self.gamma,
                "degree" : self.degree,
                "coef0" : self.coef0}

    def fit(self, X, y):
        param_name, param_value = _get_regularization_param(self.estimator_)
        return self._fit_one(X, y, param_value, self._kernel_params())

    def predict(self, X):
        K = pairwise_kernels(X, self.dictionary_, metric=self.metric,
                             filter_params=True, **self._kernel_params())
        return self.estimator_.predict(K)


class PrimalClassifierCV(BasePrimal, ClassifierMixin):

    def __init__(self,
                 # base_estimator
                 estimator,
                 # cv
                 param_grid=None,
                 cv=3,
                 upper_bound=None,
                 # dictionary
                 dictionary_size=None, trim_dictionary=True,
                 # learning
                 debiasing=False,
                 # metric
                 metric="linear",
                 # misc
                 random_state=None, verbose=0):

        self.estimator_ = estimator
        self.cv = cv
        self.param_grid = param_grid
        self.upper_bound = upper_bound
        self.dictionary_size = dictionary_size
        self.trim_dictionary = trim_dictionary
        self.debiasing = debiasing
        self.metric = metric
        self.random_state = random_state
        self.verbose = verbose

        self._set_params()

    def _set_params(self):
        if self.param_grid is None:
            raise AttributeError("Missing param_grid.")

        self.regul_name = None
        for param in ("C", "estimator__C", "alpha", "estimator__alpha"):
            if param in self.param_grid:
                self.regul_name = param
                break

        if self.regul_name is None:
            raise AttributeError("Missing regularization parameter.")

        self.regul_values = np.sort(self.param_grid[self.regul_name])

        if "alpha" in self.regul_name:
            self.regul_values = self.regul_values[::-1]

        del self.param_grid[self.regul_name]

    def _fit_path(self, X_train, y_train, X_val, y_val, kernel_params):
        random_state = check_random_state(self.random_state)

        if self.verbose: print "Creating dictionary..."
        dictionary = _dictionary(X_train, self.dictionary_size, random_state)

        if self.verbose: print "Computing kernel..."
        K_train = pairwise_kernels(X_train, dictionary, metric=self.metric,
                                   filter_params=True, **kernel_params)

        K_val = pairwise_kernels(X_val, dictionary, metric=self.metric,
                                 filter_params=True, **kernel_params)

        scores = []
        estimators = []
        n_svs = []
        estimator = clone(self.estimator_)

        for param in self.regul_values:
            if self.verbose: print "Computing model for %s..." % str(param)
            estimator.set_params(**{self.regul_name: param})
            estimator.fit(K_train, y_train)

            nsv = np.mean(np.sum(estimator.coef_ != 0, axis=1))
            if nsv == 0:
                score = 0
            else:
                score = estimator.score(K_val, y_val)
            # FIXME: we could stop here if nsv > upper_bound
            scores.append(score)
            estimators.append(estimator)
            n_svs.append(nsv)

        # Use the dictionary size if upper_bound is not provided
        upper_bound = self.upper_bound
        if upper_bound is None:
            upper_bound = dictionary.shape[0]

        # Need to scale down upper_bound to account for the fact
        # that we are working on a smaller part of the dataset.
        n_train = X_train.shape[0]
        n_val = X_val.shape[0]
        train_prop = n_train / float(n_train + n_val)

        scores = np.array(scores)
        n_svs = np.array(n_svs) / upper_bound * train_prop

        return scores, n_svs

    def fit(self, X, y):
        cv = list(check_cv(self.cv, X, y, is_classifier(self.estimator_)))
        n_folds = len(cv)

        if len(self.param_grid) == 0:
            grid = ({},)
        else:
            grid = list(IterGrid(self.param_grid))

        scores = []
        n_svs = []
        best_score = -np.inf
        best_params = None

        for kernel_params in grid:
            if self.verbose: print "Training: ", kernel_params
            scores = None
            n_svs = None

            i = 1
            for train, val in cv:
                if self.verbose: print "Fold %d/%d" % (i, n_folds)
                ret = self._fit_path(X[train], y[train],
                                     X[val], y[val],
                                     kernel_params)
                if scores is None:
                    scores, n_svs = ret
                else:
                    scores += ret[0]
                    n_svs += ret[1]

                i += 1

            scores /= n_folds
            n_svs /= n_folds

            scores[n_svs > 1.0] = -np.inf
            best = scores.argmax()
            if scores[best] > best_score:
                best_score = scores[best]
                best_params = (self.regul_values[best], kernel_params)

        if best_params is None:
            raise ValueError("Could not find a parameters below upper bound")

        self.regul_value_, self.kernel_params_ = best_params
        # FIXME: don't retrain if there was only one fold
        self._fit_one(X, y, self.regul_value_, self.kernel_params_)

        nsv = np.mean(np.sum(self.estimator_.coef_ != 0, axis=1))

        if nsv == 0:
            raise ValueError("Training failed...")

        return self

    def predict(self, X):
        K = pairwise_kernels(X, self.dictionary_, metric=self.metric,
                             filter_params=True, **self.kernel_params_)
        return self.estimator_.predict(K)
