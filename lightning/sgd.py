# Author: Mathieu Blondel
# License: BSD

import warnings

import numpy as np

from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin
from sklearn.utils import check_random_state
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.validation import assert_all_finite

from .base import BaseClassifier
from .base import BaseRegressor

from .sgd_fast import _binary_sgd
from .sgd_fast import _multiclass_sgd

from .sgd_fast import ModifiedHuber
from .sgd_fast import Hinge
from .sgd_fast import SquaredHinge
from .sgd_fast import Log
from .sgd_fast import SparseLog
from .sgd_fast import SquaredLoss
from .sgd_fast import Huber
from .sgd_fast import EpsilonInsensitive

from .sgd_fast import MulticlassLog
from .sgd_fast import MulticlassHinge
from .sgd_fast import MulticlassSquaredHinge


class BaseSGD(object):

    def _get_penalty(self):
        penalties = {
            "nn" : -1,
            "nnl1" : -1,
            "nnl2" : -2,
            "l1" : 1,
            "l2" : 2,
            "l1/l2" : 12
        }
        return penalties[self.penalty]

    def _get_learning_rate(self):
        learning_rates = {"constant": 1, "pegasos": 2, "invscaling": 3}
        return learning_rates[self.learning_rate]


class SGDClassifier(BaseClassifier, ClassifierMixin, BaseSGD):

    def __init__(self, loss="hinge", penalty="l2",
                 multiclass=False, alpha=0.01,
                 kernel="linear", gamma=0.1, coef0=1, degree=4,
                 learning_rate="pegasos", eta0=0.03, power_t=0.5,
                 epsilon=0.01, fit_intercept=True, intercept_decay=1.0,
                 n_components=0, max_iter=10, shuffle=True, random_state=None,
                 callback=None, n_calls=100,
                 cache_mb=500, verbose=0, n_jobs=1):
        self.loss = loss
        self.penalty = penalty
        self.multiclass = multiclass
        self.alpha = alpha
        self.kernel = kernel
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.power_t = power_t
        self.epsilon = epsilon
        self.fit_intercept = fit_intercept
        self.intercept_decay = intercept_decay
        self.n_components = n_components
        self.max_iter = max_iter
        self.shuffle = shuffle
        self.random_state = random_state
        self.callback = callback
        self.n_calls = n_calls
        self.cache_mb = cache_mb
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.coef_ = None
        self.support_vectors_ = None

    def _get_loss(self):
        if self.multiclass == True:
            losses = {
                "log" : MulticlassLog(),
                "hinge" : MulticlassHinge(),
                "squared_hinge" : MulticlassSquaredHinge(),
            }
        else:
            losses = {
                "modified_huber" : ModifiedHuber(),
                "hinge" : Hinge(1.0),
                "squared_hinge" : SquaredHinge(1.0),
                "perceptron" : Hinge(0.0),
                "log": Log(),
                "sparse_log" : SparseLog(),
                "squared" : SquaredLoss(),
                "huber" : Huber(self.epsilon),
                "epsilon_insensitive" : EpsilonInsensitive(self.epsilon)
            }
        return losses[self.loss]


    def fit(self, X, y):
        rs = check_random_state(self.random_state)

        reencode = self.multiclass == True
        y, n_classes, n_vectors = self._set_label_transformers(y, reencode)

        kernel = False if self.kernel == "linear" else True
        ds = self._get_dataset(X, kernel=kernel)
        n_samples = ds.get_n_samples()
        n_features = ds.get_n_features()
        d = n_features if self.kernel == "linear" else n_samples
        self.coef_ = np.zeros((n_vectors, d), dtype=np.float64)

        self.intercept_ = np.zeros(n_vectors, dtype=np.float64)

        loss = self._get_loss()
        penalty = self._get_penalty()
        eta0 = self.eta0
        if self.learning_rate == "invscaling" and self.power_t == 0.5 and \
           self.eta0 == "auto":
               D = loss.max_diameter(ds, n_vectors, penalty, self.alpha)
               G = loss.max_gradient(ds, n_vectors)
               eta0 = D / (4.0 * G)
               if self.verbose >= 1:
                   print "eta0=%f" % eta0

        if n_vectors == 1 or self.multiclass == False:
            Y = np.asfortranarray(self.label_binarizer_.fit_transform(y),
                                  dtype=np.float64)
            for i in xrange(n_vectors):
                _binary_sgd(self,
                            self.coef_, self.intercept_, i,
                            ds, Y[:, i], loss, penalty,
                            self.n_components, self.alpha,
                            self._get_learning_rate(),
                            eta0, self.power_t,
                            self.fit_intercept,
                            self.intercept_decay,
                            int(self.max_iter * n_samples), self.shuffle, rs,
                            self.callback, self.n_calls, self.verbose)

        elif self.multiclass == True:
            _multiclass_sgd(self, self.coef_, self.intercept_,
                 ds, y.astype(np.int32), loss, penalty,
                 self.n_components, self.alpha, self._get_learning_rate(),
                 eta0, self.power_t, self.fit_intercept, self.intercept_decay,
                 int(self.max_iter * n_samples), self.shuffle, rs,
                 self.callback, self.n_calls, self.verbose)

        else:
            raise ValueError("Wrong value for multiclass.")

        try:
            assert_all_finite(self.coef_)
        except ValueError:
            warnings.warn("coef_ contains infinite values")

        if self.kernel != "linear":
            self._post_process(X)

        return self

    def decision_function(self, X):
        kernel = False if self.kernel == "linear" else True
        ds = self._get_dataset(X, self.support_vectors_, kernel=kernel)
        return ds.dot(self.coef_.T) + self.intercept_


class SGDRegressor(BaseRegressor, RegressorMixin, BaseSGD):

    def __init__(self, loss="squared", penalty="l2",
                 alpha=0.01,
                 kernel="linear", gamma=0.1, coef0=1, degree=4,
                 learning_rate="pegasos", eta0=0.03, power_t=0.5,
                 epsilon=0.01, fit_intercept=True, intercept_decay=1.0,
                 n_components=0, max_iter=10, shuffle=True, random_state=None,
                 callback=None, n_calls=100,
                 cache_mb=500, verbose=0, n_jobs=1):
        self.loss = loss
        self.penalty = penalty
        self.alpha = alpha
        self.kernel = kernel
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.power_t = power_t
        self.epsilon = epsilon
        self.fit_intercept = fit_intercept
        self.intercept_decay = intercept_decay
        self.n_components = n_components
        self.max_iter = max_iter
        self.shuffle = shuffle
        self.random_state = random_state
        self.callback = callback
        self.n_calls = n_calls
        self.cache_mb = cache_mb
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.coef_ = None
        self.support_vectors_ = None

    def _get_loss(self):
        losses = {
            "squared" : SquaredLoss(),
            "huber" : Huber(self.epsilon),
            "epsilon_insensitive" : EpsilonInsensitive(self.epsilon)
        }
        return losses[self.loss]


    def fit(self, X, y):
        rs = check_random_state(self.random_state)

        kernel = False if self.kernel == "linear" else True
        ds = self._get_dataset(X, kernel=kernel)
        n_samples = ds.get_n_samples()
        n_features = ds.get_n_features()
        d = n_features if self.kernel == "linear" else n_samples

        self.outputs_2d_ = len(y.shape) == 2
        if self.outputs_2d_:
            Y = y
        else:
            Y = y.reshape(-1, 1)
        Y = np.asfortranarray(Y)
        n_vectors = Y.shape[1]
        self.coef_ = np.zeros((n_vectors, d), dtype=np.float64)
        self.intercept_ = np.zeros(n_vectors, dtype=np.float64)

        loss = self._get_loss()
        penalty = self._get_penalty()
        eta0 = self.eta0

        for k in xrange(n_vectors):
            _binary_sgd(self,
                        self.coef_, self.intercept_, k,
                        ds, Y[:, k], loss, penalty,
                        self.n_components, self.alpha,
                        self._get_learning_rate(),
                        eta0, self.power_t,
                        self.fit_intercept,
                        self.intercept_decay,
                        int(self.max_iter * n_samples), self.shuffle, rs,
                        self.callback, self.n_calls, self.verbose)


        try:
            assert_all_finite(self.coef_)
        except ValueError:
            warnings.warn("coef_ contains infinite values")

        if self.kernel != "linear":
            self._post_process(X)

        return self

    def predict(self, X):
        try:
            assert_all_finite(self.coef_)
            pred = safe_sparse_dot(X, self.coef_.T)
        except ValueError:
            n_samples = X.shape[0]
            n_vectors = self.coef_.shape[0]
            pred = np.zeros((n_samples, n_vectors))

        if not self.outputs_2d_:
            pred = pred.ravel()

        return pred
