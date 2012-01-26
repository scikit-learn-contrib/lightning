# Author: Mathieu Blondel
# License: BSD

import time

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import LabelBinarizer, Scaler
from sklearn.utils import check_random_state
from sklearn.externals.joblib import Parallel, delayed


from .primal import _dictionary as _components


class SquaredLoss(object):

    def negative_gradient(self, y_true, y_pred):
        return y_true - y_pred

    def line_search(self, y, y_pred, column):
        squared_norm = np.sum(column ** 2)
        residuals = y - y_pred
        return np.dot(column, residuals) / squared_norm


def _fit_generator(estimator, loss, K, y, n_nonzero_coefs, norms,
                   n_refit, check_duplicates, verbose=0):
    n_samples = K.shape[0]
    n_components = K.shape[1]
    coef = np.zeros(n_components, dtype=np.float64)
    selected = np.zeros(n_components, dtype=bool)
    y_pred = np.zeros(n_samples, dtype=np.float64)

    if loss is None:
        residuals = y.copy()

    for i in range(1, n_nonzero_coefs + 1):

        if verbose >= 2: print "Learning %d/%d..." % (i, n_nonzero_coefs)

        # compute pseudo-residuals if needed
        if loss is not None:
            residuals = loss.negative_gradient(y, y_pred)

        # select best basis
        dots = np.dot(K.T, residuals)
        dots /= norms
        abs_dots = np.abs(dots)
        if check_duplicates:
            abs_dots[selected] = -np.inf
        best = np.argmax(abs_dots)
        selected[best] = True

        if n_refit != 0 and i % n_refit == 0:
            # fit the selected coefficient and the previous ones too
            K_subset = K[:, selected]
            estimator.fit(K_subset, y)
            coef[selected] = estimator.coef_.ravel()

            y_pred = estimator.decision_function(K_subset)

            if loss is None:
                residuals = y - y_pred

            refitted = True
        else:
            # find coefficient for the selected basis only
            if loss is None:
                alpha =  dots[best] / norms[best]
            else:
                alpha = loss.line_search(y, y_pred, K[:, best])

            coef[best] += alpha
            weighted_basis = alpha * K[:, best]

            y_pred += weighted_basis

            if loss is None:
                residuals -= weighted_basis

            refitted = False

        yield coef, y_pred

    # fit one last time
    #K_subset = K[:, selected]
    #estimator.fit(K_subset, y)
    #coef[selected] = estimator.coef_.ravel()
    #yield coef


def _fit_last(estimator, loss, K, y, n_nonzero_coefs, norms,
              n_refit, check_duplicates):

    for coef, y_pred in _fit_generator(estimator, loss, K, y,
                                       n_nonzero_coefs, norms,
                                       n_refit, check_duplicates):
        pass

    return coef


class KMPBase(BaseEstimator):

    def __init__(self,
                 n_nonzero_coefs=0.3,
                 loss=None,
                 # components (basis functions)
                 n_components=None,
                 check_duplicates=False,
                 scale=False,
                 # back-fitting
                 n_refit=5,
                 estimator=None,
                 # metric
                 metric="linear", gamma=0.1, coef0=1, degree=4,
                 # validation
                 X_val=None, y_val=None,
                 n_validate=1,
                 # misc
                 random_state=None, verbose=0, n_jobs=1):
        if n_nonzero_coefs < 0:
            raise AttributeError("n_nonzero_coefs should be > 0.")

        self.n_nonzero_coefs = n_nonzero_coefs
        self.loss = loss
        self.n_components = n_components
        self.check_duplicates = check_duplicates
        self.scale = scale
        self.n_refit = n_refit
        self.estimator = estimator
        self.metric = metric
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
        self.X_val = X_val
        self.y_val = y_val
        self.n_validate = n_validate
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

    def _pref_fit(self, X, y):
        random_state = check_random_state(self.random_state)

        n_nonzero_coefs = self.n_nonzero_coefs
        if 0 < n_nonzero_coefs and n_nonzero_coefs <= 1:
            n_nonzero_coefs = int(n_nonzero_coefs * X.shape[0])

        if self.verbose: print "Creating components..."
        components = _components(X, self.n_components, random_state)

        if n_nonzero_coefs > components.shape[0]:
            raise AttributeError("n_nonzero_coefs cannot be bigger than "
                                 "n_components.")

        if self.verbose: print "Computing dictionary..."
        start = time.time()
        K = pairwise_kernels(X, components, metric=self.metric,
                             filter_params=True, n_jobs=self.n_jobs,
                             **self._kernel_params())
        if self.verbose: print "Done in", time.time() - start, "seconds"

        if self.scale:
            if self.verbose: print "Scaling dictionary"
            start = time.time()
            self.scaler_ = Scaler(copy=False)
            K = self.scaler_.fit_transform(K)
            if self.verbose: print "Done in", time.time() - start, "seconds"


        # FIXME: this allocates a lot of intermediary memory
        norms = np.sqrt(np.sum(K ** 2, axis=0))

        self.components_ = components

        return n_nonzero_coefs, K, norms

    def _fit_multi(self, K, y, Y, n_nonzero_coefs, norms):
        if self.verbose: print "Starting training..."
        start = time.time()
        coef = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                delayed(_fit_last)(self._get_estimator(), self._get_loss(),
                                   K, Y[:, i], n_nonzero_coefs, norms,
                                   self.n_refit, self.check_duplicates)
                for i in xrange(Y.shape[1]))
        self.coef_ = np.array(coef)
        if self.verbose: print "Done in", time.time() - start, "seconds"

    def _fit_multi_with_validation(self, K, y, Y, n_nonzero_coefs, norms):
        iterators = [_fit_generator(self._get_estimator(), self._get_loss(),
                                    K, Y[:, i], n_nonzero_coefs, norms,
                                    self.n_refit, self.check_duplicates,
                                    self.verbose)
                     for i in xrange(Y.shape[1])]

        if self.verbose: print "Computing validation dictionary..."
        start = time.time()
        K_val = pairwise_kernels(self.X_val, self.components_,
                                 metric=self.metric,
                                 filter_params=True,
                                 n_jobs=self.n_jobs,
                                 **self._kernel_params())
        if self.verbose: print "Done in", time.time() - start, "seconds"
        if self.scale:
            K_val = self.scaler_.transform(K_val)


        if self.verbose: print "Starting training..."
        start = time.time()
        best_score = -np.inf
        validation_scores = []
        training_scores = []
        iterations = []
        try:
            n_iter = 1
            while True:
                res = [it.next() for it in iterators]
                coef, y_train = zip(*res)
                coef = np.array(coef)
                y_train = np.array(y_train).T

                if n_iter % self.n_validate == 0:
                    if self.verbose >= 2:
                        print "Validating %d/%d..." % (n_iter, n_nonzero_coefs)
                    y_val = np.dot(K_val, coef.T)

                    if hasattr(self, "lb_"):
                        y_val = self.lb_.inverse_transform(y_val,
                                                           threshold=0.5)
                        validation_score = np.mean(y_val == self.y_val)
                        y_train = self.lb_.inverse_transform(y_train,
                                                             threshold=0.5)
                        training_score = np.mean(y_train == y)
                    else:
                        validation_score = -np.mean((y_val - self.y_val) ** 2)
                        training_score = -np.mean((y_train - Y) ** 2)

                    if validation_score > best_score:
                        self.coef_ = coef.copy()
                        best_score = validation_score

                    validation_scores.append(validation_score)
                    training_scores.append(training_score)
                    iterations.append(n_iter)
                n_iter += 1
        except StopIteration:
            pass

        self.validation_scores_ = np.array(validation_scores)
        self.training_scores_ = np.array(training_scores)
        self.iterations_ = np.array(iterations)

        if self.verbose: print "Done in", time.time() - start, "seconds"

    def _fit(self, K, y, Y, n_nonzero_coefs, norms):
        if self.X_val is not None and self.y_val is not None:
            meth = self._fit_multi_with_validation
        else:
            meth = self._fit_multi
        meth(K, y, Y, n_nonzero_coefs, norms)

    def _post_fit(self):
        used_basis = np.sum(self.coef_ != 0, axis=0, dtype=bool)
        self.coef_ = self.coef_[:, used_basis]
        self.components_ = self.components_[used_basis]

    def decision_function(self, X):
        K = pairwise_kernels(X, self.components_, metric=self.metric,
                             filter_params=True, n_jobs=self.n_jobs,
                             **self._kernel_params())
        if self.scale:
            K = self.scaler_.transform(K)
        return np.dot(K, self.coef_.T)


class KMPClassifier(KMPBase, ClassifierMixin):

    def fit(self, X, y):
        n_nonzero_coefs, K, norms = self._pref_fit(X, y)

        self.lb_ = LabelBinarizer()
        Y = self.lb_.fit_transform(y)
        self._fit(K, y, Y, n_nonzero_coefs, norms)

        self._post_fit()

        return self

    def predict(self, X):
        pred = self.decision_function(X)
        return self.lb_.inverse_transform(pred, threshold=0.5)


class KMPRegressor(KMPBase, RegressorMixin):

    def fit(self, X, y):
        n_nonzero_coefs, K, norms = self._pref_fit(X, y)

        Y = y.reshape(-1, 1) if len(y.shape) == 1 else y
        self._fit(K, y, Y, n_nonzero_coefs, norms)

        self._post_fit()

        return self

    def predict(self, X):
        return self.decision_function(X)

