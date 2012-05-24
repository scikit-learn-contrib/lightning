# Author: Mathieu Blondel
# License: BSD

import numpy as np

from sklearn.base import ClassifierMixin, clone
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.utils import check_random_state

from .base import BaseLinearClassifier, BaseKernelClassifier
from .kernel_fast import get_kernel, KernelCache

from .sgd_fast import _binary_sgd
from .sgd_fast import _multiclass_hinge_sgd
from .sgd_fast import _multiclass_log_sgd

from .sgd_fast import ModifiedHuber
from .sgd_fast import Hinge
from .sgd_fast import Log
from .sgd_fast import SparseLog
from .sgd_fast import SquaredLoss
from .sgd_fast import Huber
from .sgd_fast import EpsilonInsensitive


class BaseSGD(object):

    def _get_loss(self):
        losses = {
            "modified_huber" : ModifiedHuber(),
            "hinge" : Hinge(1.0),
            "perceptron" : Hinge(0.0),
            "log": Log(),
            "sparse_log" : SparseLog(),
            "squared" : SquaredLoss(),
            "huber" : Huber(self.epsilon),
            "epsilon_insensitive" : EpsilonInsensitive(self.epsilon)
        }
        return losses[self.loss]

    def _get_learning_rate(self):
        learning_rates = {"constant": 1, "pegasos": 2, "invscaling": 3}
        return learning_rates[self.learning_rate]

    def _set_label_transformers(self, y):
        if self.multiclass == "natural":
            self.label_encoder_ = LabelEncoder()
            y = self.label_encoder_.fit_transform(y)

        self.label_binarizer_ = LabelBinarizer(neg_label=-1, pos_label=1)
        self.label_binarizer_.fit(y)
        self.classes_ = self.label_binarizer_.classes_.astype(np.int32)
        n_classes = len(self.label_binarizer_.classes_)
        n_vectors = 1 if n_classes <= 2 else n_classes
        return n_classes, n_vectors

class SGDClassifier(BaseLinearClassifier, BaseSGD, ClassifierMixin):

    def __init__(self, loss="hinge", multiclass="one-vs-rest", lmbda=0.01,
                 learning_rate="pegasos", eta0=0.03, power_t=0.5,
                 epsilon=0.01, fit_intercept=True, intercept_decay=1.0,
                 max_iter=10, random_state=None, verbose=0, n_jobs=1):
        self.loss = loss
        self.multiclass = multiclass
        self.lmbda = lmbda
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.power_t = power_t
        self.epsilon = epsilon
        self.fit_intercept = fit_intercept
        self.intercept_decay = intercept_decay
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.coef_ = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        rs = check_random_state(self.random_state)

        n_classes, n_vectors = self._set_label_transformers(y)

        self.coef_ = np.zeros((n_vectors, n_features), dtype=np.float64)
        self.intercept_ = np.zeros(n_vectors, dtype=np.float64)

        kernel = get_kernel("linear")
        kcache = KernelCache(kernel, n_samples, 0, 0, self.verbose)

        if n_vectors == 1 or self.multiclass == "one-vs-rest":
            Y = self.label_binarizer_.transform(y)
            for i in xrange(n_vectors):
                _binary_sgd(self,
                            self.coef_, self.intercept_, i,
                            X, Y[:, i],
                            self._get_loss(),
                            kcache, 1, 0,
                            self.lmbda,
                            self._get_learning_rate(),
                            self.eta0, self.power_t,
                            self.fit_intercept,
                            self.intercept_decay,
                            self.max_iter * n_samples,
                            rs, self.verbose)

        elif self.multiclass == "natural":
            if self.loss in ("hinge", "log"):
                func = eval("_multiclass_%s_sgd" % self.loss)
                func(self, self.coef_, self.intercept_,
                     X, y.astype(np.int32), kcache, 1, 0, self.lmbda,
                     self._get_learning_rate(), self.eta0, self.power_t,
                     self.fit_intercept, self.intercept_decay,
                     self.max_iter * n_samples, rs, self.verbose)
            else:
                raise ValueError("Loss not supported for multiclass!")

        else:
            raise ValueError("Wrong value for multiclass.")

        return self


class KernelSGDClassifier(BaseKernelClassifier, BaseSGD, ClassifierMixin):

    def __init__(self, loss="hinge", multiclass="one-vs-rest", lmbda=0.01,
                 kernel="linear", gamma=0.1, coef0=1, degree=4,
                 learning_rate="pegasos", eta0=0.03, power_t=0.5,
                 epsilon=0.01, fit_intercept=True, intercept_decay=1.0,
                 n_components=0, max_iter=10, random_state=None,
                 cache_mb=500, verbose=0, n_jobs=1):
        self.loss = loss
        self.multiclass = multiclass
        self.lmbda = lmbda
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
        self.random_state = random_state
        self.cache_mb = cache_mb
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.coef_ = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        rs = check_random_state(self.random_state)

        n_classes, n_vectors = self._set_label_transformers(y)

        self.coef_ = np.zeros((n_vectors, n_samples), dtype=np.float64)
        self.intercept_ = np.zeros(n_vectors, dtype=np.float64)

        kernel = self._get_kernel()
        kcache = KernelCache(kernel, n_samples,
                             self.cache_mb, 1, self.verbose)

        if n_vectors == 1 or self.multiclass == "one-vs-rest":
            Y = self.label_binarizer_.transform(y)
            for i in xrange(n_vectors):
                _binary_sgd(self,
                            self.coef_, self.intercept_, i,
                            X, Y[:, i],
                            self._get_loss(),
                            kcache, 0, self.n_components,
                            self.lmbda,
                            self._get_learning_rate(),
                            self.eta0, self.power_t,
                            self.fit_intercept,
                            self.intercept_decay,
                            self.max_iter * n_samples,
                            rs, self.verbose)

        elif self.multiclass == "natural":
            if self.loss in ("hinge", "log"):
                func = eval("_multiclass_%s_sgd" % self.loss)
                func(self, self.coef_, self.intercept_,
                     X, y.astype(np.int32), kcache, 0, self.n_components,
                     self.lmbda, self._get_learning_rate(), self.eta0,
                     self.power_t, self.fit_intercept, self.intercept_decay,
                     self.max_iter * n_samples, rs, self.verbose)
            else:
                raise ValueError("Loss not supported for multiclass!")

        else:
            raise ValueError("Wrong value for multiclass.")

        self._post_process(X)

        return self
