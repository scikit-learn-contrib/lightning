# Author: Mathieu Blondel
# License: BSD

import numpy as np
import scipy.sparse as sp

from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_random_state

from .base import BaseClassifier
from .base import BaseRegressor

from .dataset_fast import KernelDataset
from .primal_cd_fast import _primal_cd

from .primal_cd_fast import Squared
from .primal_cd_fast import SquaredHinge
from .primal_cd_fast import Squared01
from .primal_cd_fast import SquaredHinge01
from .primal_cd_fast import ModifiedHuber
from .primal_cd_fast import Log


class BaseCD(object):

    def _get_loss(self):
        params = {"max_steps" : self.max_steps,
                  "sigma" : self.sigma,
                  "beta" : self.beta,
                  "verbose" : self.verbose}
        losses = {
            "squared" : Squared(verbose=self.verbose),
            "squared_hinge" : SquaredHinge(**params),
            "modified_huber" : ModifiedHuber(**params),
            "log" : Log(**params),
        }

        #if self.penalty == "nn":
            #losses["squared"] = Squared01(verbose=self.verbose)
            #losses["squared_hinge"] = SquaredHinge01(**params)

        return losses[self.loss]

    def _get_penalty(self):
        penalties = {
            "l1" : 1,
            "l2" : 2,
            "nn" : -1,
            "nnl1" : -1,
            "nnl2" : -2
        }
        return penalties[self.penalty]

    def _init_errors(self, Y):
        n_samples, n_vectors = Y.shape
        if self.loss == "squared":
            self.errors_ = -Y.T
        #elif self.loss == "squared_hinge" and self.penalty == "nn":
            #self.errors_ = 2 * Y.T
        else:
            self.errors_ = np.ones((n_vectors, n_samples), dtype=np.float64)


class CDClassifier(BaseCD, BaseClassifier, ClassifierMixin):

    def __init__(self, C=1.0, alpha=1.0, U=1e12,
                 loss="squared_hinge", penalty="l2",
                 multiclass=False,
                 max_iter=50, tol=1e-3, termination="violation_sum",
                 shrinking=True,
                 max_steps=30, sigma=0.01, beta=0.5,
                 kernel=None, gamma=0.1, coef0=1, degree=4, cache_mb=500,
                 warm_start=False, debiasing=False, Cd=1.0,
                 warm_debiasing=False,
                 selection="cyclic", search_size=60, permute=True,
                 n_components=1000, components=None,
                 callback=None, n_calls=100,
                 random_state=None, verbose=0, n_jobs=1):
        self.C = C
        self.alpha = alpha
        self.U = U
        self.loss = loss
        self.penalty = penalty
        self.multiclass = multiclass
        self.max_iter = max_iter
        self.tol = tol
        self.termination = termination
        self.shrinking = shrinking
        self.max_steps = max_steps
        self.sigma = sigma
        self.beta = beta
        self.kernel = kernel
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
        self.cache_mb = cache_mb
        self.warm_start = warm_start
        self.debiasing = debiasing
        self.Cd = Cd
        self.warm_debiasing = warm_debiasing
        self.selection = selection
        self.search_size = search_size
        self.permute = permute
        self.n_components = n_components
        self.components = components
        self.callback = callback
        self.n_calls = n_calls
        self.random_state = random_state
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.support_vectors_ = None
        self.coef_ = None
        self.violation_init_ = {}

    def fit(self, X, y):
        rs = self._get_random_state()

        # Create dataset
        ds = self._get_dataset(X, self.components, order="fortran")
        n_samples = ds.get_n_samples()
        n_features = ds.get_n_features()

        # Create label transformers
        #neg_label = 0 if self.penalty == "nn" else -1
        reencode = self.penalty == "l1/l2"
        y, n_classes, n_vectors = \
                self._set_label_transformers(y, reencode, neg_label=-1)
        Y = np.asfortranarray(self.label_binarizer_.transform(y),
                              dtype=np.float64)

        # Initialize coefficients
        if self.warm_start and self.coef_ is not None:
            if self.kernel:
                coef = np.zeros((n_vectors, n_features), dtype=np.float64)
                coef[:, self.support_indices_] = self.coef_
                self.coef_ = coef
        else:
            self.C_init = self.C
            self.coef_ = np.zeros((n_vectors, n_features), dtype=np.float64)
            self._init_errors(Y)


        self.intercept_ = np.zeros(n_vectors, dtype=np.float64)
        indices = np.arange(n_features, dtype=np.int32)

        # Learning
        if self.penalty == "l1/l2":
            tol = self.tol
            #n_min = np.min(np.sum(Y == 1, axis=0))
            #tol *= max(n_min, 1) / n_samples

            vinit = self.violation_init_.get(0, 0) * self.C / self.C_init
            viol = _primal_cd(self, self.coef_, self.errors_,
                              ds, y, Y, -1, self.multiclass,
                              indices, 12, self._get_loss(),
                              self.selection, self.search_size, self.permute,
                              self.termination, self.n_components,
                              self.C, self.alpha, self.U,
                              self.max_iter, self.max_steps,
                              self.shrinking, vinit,
                              rs, tol, self.callback, self.n_calls,
                              self.verbose)
            if self.warm_start and len(self.violation_init_) == 0:
                self.violation_init_[0] = viol

        elif self.penalty in ("l1", "l2", "nn"):
            penalty = self._get_penalty()
            for k in xrange(n_vectors):
                n_pos = np.sum(Y[:, k] == 1)
                n_neg = n_samples - n_pos
                tol = self.tol * max(min(n_pos, n_neg), 1) / n_samples

                vinit = self.violation_init_.get(k, 0) * self.C / self.C_init
                viol = _primal_cd(self, self.coef_, self.errors_,
                                  ds, y, Y, k, False,
                                  indices, penalty, self._get_loss(),
                                  self.selection, self.search_size,
                                  self.permute,
                                  self.termination, self.n_components,
                                  self.C, self.alpha, self.U,
                                  self.max_iter, self.max_steps,
                                  self.shrinking, vinit,
                                  rs, tol, self.callback, self.n_calls,
                                  self.verbose)

                if self.warm_start and not k in self.violation_init_:
                    self.violation_init_[k] = viol

        if self.debiasing:
            nz = np.sum(self.coef_ != 0, axis=0, dtype=bool)
            self.support_indices_ = np.arange(n_features, dtype=np.int32)[nz]
            indices = self.support_indices_.copy()
            if not self.warm_debiasing:
                self.coef_ = np.zeros((n_vectors, n_features), dtype=np.float64)
                self._init_errors(Y)

            for k in xrange(n_vectors):
                _primal_cd(self, self.coef_, self.errors_,
                           ds, y, Y, k, False,
                           indices, 2, self._get_loss(),
                           "cyclic", self.search_size, self.permute,
                           "violation_sum", self.n_components,
                           self.Cd, 1.0, self.U,
                           self.max_iter, self.max_steps,
                           self.shrinking, 0,
                           rs, self.tol, self.callback, self.n_calls,
                           self.verbose)

        nz = np.sum(self.coef_ != 0, axis=0, dtype=bool)
        self.support_indices_ = np.arange(n_features, dtype=np.int32)[nz]

        if self.kernel:
            A = X if self.components is None else self.components
            self._post_process(A)

        return self

    def decision_function(self, X):
        ds = self._get_dataset(X, self.support_vectors_)
        return ds.dot(self.coef_.T) + self.intercept_


class CDRegressor(BaseCD, BaseRegressor, RegressorMixin):

    def __init__(self, C=1.0, alpha=1.0, U=1e12,
                 loss="squared", penalty="l2",
                 max_iter=50, tol=1e-3, termination="violation_sum",
                 shrinking=True,
                 max_steps=30, sigma=0.01, beta=0.5,
                 kernel=None, gamma=0.1, coef0=1, degree=4, cache_mb=500,
                 warm_start=False, debiasing=False, Cd=1.0,
                 warm_debiasing=False,
                 selection="cyclic", permute=True,
                 n_components=1000, components=None,
                 callback=None, n_calls=100,
                 random_state=None, verbose=0, n_jobs=1):
        self.C = C
        self.alpha = alpha
        self.U = U
        self.loss = loss
        self.penalty = penalty
        self.max_iter = max_iter
        self.tol = tol
        self.termination = termination
        self.shrinking = shrinking
        self.max_steps = max_steps
        self.sigma = sigma
        self.beta = beta
        self.kernel = kernel
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
        self.cache_mb = cache_mb
        self.warm_start = warm_start
        self.debiasing = debiasing
        self.Cd = Cd
        self.warm_debiasing = warm_debiasing
        self.selection = selection
        self.permute = permute
        self.n_components = n_components
        self.components = components
        self.callback = callback
        self.n_calls = n_calls
        self.random_state = random_state
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.support_vectors_ = None
        self.coef_ = None
        self.violation_init_ = {}

    def fit(self, X, y):
        rs = self._get_random_state()

        # Create dataset
        ds = self._get_dataset(X, self.components, order="fortran")
        n_samples = ds.get_n_samples()
        n_features = ds.get_n_features()

        self.outputs_2d_ = len(y.shape) == 2
        if self.outputs_2d_:
            Y = y
        else:
            Y = y.reshape(-1, 1)
        Y = np.asfortranarray(Y, dtype=np.float64)
        y = np.empty(0, dtype=np.int32)
        n_vectors = Y.shape[1]

        # Initialize coefficients
        if self.warm_start and self.coef_ is not None:
            if self.kernel:
                coef = np.zeros((n_vectors, n_features), dtype=np.float64)
                coef[:, self.support_indices_] = self.coef_
                self.coef_ = coef
        else:
            self.C_init = self.C
            self.coef_ = np.zeros((n_vectors, n_features), dtype=np.float64)
            self._init_errors(Y)

        self.intercept_ = np.zeros(n_vectors, dtype=np.float64)
        indices = np.arange(n_features, dtype=np.int32)

        penalty = self._get_penalty()
        for k in xrange(n_vectors):
            vinit = self.violation_init_.get(k, 0) * self.C / self.C_init
            viol = _primal_cd(self, self.coef_, self.errors_,
                              ds, y, Y, k, False,
                              indices, penalty, self._get_loss(),
                              self.selection, 0, self.permute,
                              self.termination, self.n_components,
                              self.C, self.alpha, self.U,
                              self.max_iter, self.max_steps,
                              self.shrinking, vinit,
                              rs, self.tol, self.callback, self.n_calls,
                              self.verbose)

            if self.warm_start and not k in self.violation_init_:
                self.violation_init_[k] = viol

        return self
