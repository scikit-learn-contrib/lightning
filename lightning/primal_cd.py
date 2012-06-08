# Author: Mathieu Blondel
# License: BSD

import numpy as np

from sklearn.base import ClassifierMixin, clone
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_random_state

from .base import BaseLinearClassifier, BaseKernelClassifier

from .kernel_fast import get_kernel, KernelCache
from .primal_cd_fast import _primal_cd_l2r
from .primal_cd_fast import _primal_cd_l2svm_l2r
from .primal_cd_fast import _primal_cd_l2svm_l1r
from .primal_cd_fast import _C_lower_bound_kernel

from .primal_cd_fast import Squared
from .primal_cd_fast import SquaredHinge
from .primal_cd_fast import ModifiedHuber
from .primal_cd_fast import Log


class BaseSVC(object):

    def _get_loss(self):
        losses = {
            "squared" : Squared(),
            "squared_hinge" : SquaredHinge(),
            "modified_huber" : ModifiedHuber(),
            "log" : Log(),
        }
        return losses[self.loss]


class PrimalLinearSVC(BaseSVC, BaseLinearClassifier, ClassifierMixin):

    def __init__(self, C=1.0, loss="squared_hinge", penalty="l2",
                 max_iter=1000, tol=1e-3,
                 termination="convergence", nz_coef_upper_bound=1000,
                 warm_start=False, random_state=None,
                 callback=None, verbose=0, n_jobs=1):
        self.C = C
        self.loss = loss
        self.penalty = penalty
        self.max_iter = max_iter
        self.tol = tol
        self.termination = termination
        self.nz_coef_upper_bound = nz_coef_upper_bound
        self.warm_start = warm_start
        self.random_state = random_state
        self.callback = callback
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.coef_ = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        rs = check_random_state(self.random_state)

        X = np.asfortranarray(X, dtype=np.float64)

        self.label_binarizer_ = LabelBinarizer(neg_label=-1, pos_label=1)
        Y = self.label_binarizer_.fit_transform(y)
        self.classes_ = self.label_binarizer_.classes_
        n_vectors = Y.shape[1]

        kernel = get_kernel("linear")
        kcache = KernelCache(kernel, n_samples, 0, 0, self.verbose)

        if not self.warm_start or self.coef_ is None:
            self.coef_ = np.zeros((n_vectors, n_features), dtype=np.float64)
            self.errors_ = np.ones((n_vectors, n_samples), dtype=np.float64)
        self.intercept_ = 0

        indices = np.arange(n_features, dtype=np.int32)

        for i in xrange(n_vectors):
            if self.penalty == "l1":
                _primal_cd_l2svm_l1r(self, self.coef_[i], self.errors_[i],
                                     X, Y[:, i], indices, kcache, True,
                                     "permute", 60,
                                     self.termination, self.nz_coef_upper_bound,
                                     self.C, self.max_iter, rs, self.tol,
                                     self.callback, verbose=self.verbose)
            else:
                _primal_cd_l2r(self, self.coef_[i], self.errors_[i],
                               X, None, Y[:, i], indices,
                               self._get_loss(), kcache, True, False,
                               "permute", 60,
                               self.termination, self.nz_coef_upper_bound,
                               self.C, self.max_iter, rs, self.tol,
                               self.callback, verbose=self.verbose)

        return self


class PrimalSVC(BaseSVC, BaseKernelClassifier, ClassifierMixin):

    def __init__(self, C=1.0, loss="squared_hinge", penalty="l1",
                 max_iter=10, tol=1e-3, kernel_regularizer=False,
                 kernel="linear", gamma=0.1, coef0=1, degree=4,
                 Cd=1.0, warm_debiasing=False,
                 selection="permute", search_size=60,
                 termination="convergence", n_components=1000,
                 cache_mb=500, warm_start=False, random_state=None,
                 components=None, callback=None, verbose=0, n_jobs=1):
        self.C = C
        self.loss = loss
        self.penalty = penalty
        self.max_iter = max_iter
        self.tol = tol
        self.kernel_regularizer = kernel_regularizer
        self.kernel = kernel
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
        self.Cd = Cd
        self.warm_debiasing = warm_debiasing
        self.selection = selection
        self.search_size = search_size
        self.termination = termination
        self.n_components = n_components
        self.cache_mb = cache_mb
        self.warm_start = warm_start
        self.random_state = random_state
        self.components = components
        self.callback = callback
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.support_vectors_ = None
        self.coef_ = None

    def fit(self, X, y, kcache=None):
        n_samples = X.shape[0]
        rs = check_random_state(self.random_state)
        X = np.ascontiguousarray(X, dtype=np.float64)

        self.label_binarizer_ = LabelBinarizer(neg_label=-1, pos_label=1)
        Y = self.label_binarizer_.fit_transform(y)
        self.classes_ = self.label_binarizer_.classes_.astype(np.int32)
        n_vectors = Y.shape[1]

        A = X
        C = self.C
        termination = self.termination
        selection = self.selection

        if self.penalty == "l2" and self.components is not None:
            A = self.components

        if self.warm_start and self.coef_ is not None:
            coef = np.zeros((n_vectors, A.shape[0]), dtype=np.float64)
            coef[:, self.support_indices_] = self.coef_
            self.coef_ = coef
        else:
            self.coef_ = np.zeros((n_vectors, A.shape[0]), dtype=np.float64)
            self.errors_ = np.ones((n_vectors, n_samples), dtype=np.float64)

        indices = np.arange(A.shape[0], dtype=np.int32)

        if kcache is None:
            kernel = self._get_kernel()
            kcache = KernelCache(kernel, n_samples,
                                 self.cache_mb, 1, self.verbose)

        self.support_vectors_ = X
        self.intercept_ = np.zeros(n_vectors, dtype=np.float64)

        if self.penalty in ("l1", "l1l2"):
            for i in xrange(n_vectors):
                    _primal_cd_l2svm_l1r(self, self.coef_[i], self.errors_[i],
                                         X, Y[:, i], indices, kcache, False,
                                         self.selection, self.search_size,
                                         self.termination, self.n_components,
                                         self.C, self.max_iter, rs, self.tol,
                                         self.callback, verbose=self.verbose)

        if self.penalty in ("l2", "l2l2"):
            for i in xrange(n_vectors):
                _primal_cd_l2r(self, self.coef_[i], self.errors_[i],
                               X, A, Y[:, i], indices,
                               self._get_loss(), kcache, False,
                               self.kernel_regularizer,
                               self.selection, self.search_size,
                               termination, self.n_components,
                               C, self.max_iter, rs, self.tol,
                               self.callback, verbose=self.verbose)

        if self.penalty in ("l1l2", "l2l2"):
            sv = np.sum(self.coef_ != 0, axis=0, dtype=bool)
            self.support_indices_ = np.arange(n_samples, dtype=np.int32)[sv]
            indices = self.support_indices_.copy()
            A = X
            self.support_vectors_ = A
            if not self.warm_debiasing:
                self.coef_ = np.zeros((n_vectors, n_samples), dtype=np.float64)
                self.errors_ = np.ones((n_vectors, n_samples), dtype=np.float64)
            C = self.Cd
            termination = "convergence"
            selection = "permute"

            for i in xrange(n_vectors):
                _primal_cd_l2r(self, self.coef_[i], self.errors_[i],
                               X, A, Y[:, i], indices,
                               self._get_loss(), kcache, False,
                               self.kernel_regularizer,
                               selection, self.search_size,
                               termination, self.n_components,
                               C, self.max_iter, rs, self.tol,
                               self.callback, verbose=self.verbose)

        sv = np.sum(self.coef_ != 0, axis=0, dtype=bool)
        self.support_indices_ = np.arange(A.shape[0], dtype=np.int32)[sv]

        if np.sum(sv) == 0:
            # Empty model...
            return self

        self._post_process(A)

        return self


class PrimalL2SVC(BaseSVC, BaseKernelClassifier, ClassifierMixin):

    def __init__(self, C=1.0,
                 max_outer=10, max_inner=20, tol=1e-3, kernel_regularizer=False,
                 kernel="linear", gamma=0.1, coef0=1, degree=4,
                 cache_mb=500, random_state=None,
                 verbose=0, n_jobs=1):
        self.C = C
        self.max_outer = max_outer
        self.max_inner = max_inner
        self.tol = tol
        self.kernel_regularizer = kernel_regularizer
        self.kernel = kernel
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
        self.cache_mb = cache_mb
        self.random_state = random_state
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.support_vectors_ = None
        self.coef_ = None

    def fit(self, X, y, kcache=None):
        n_samples = X.shape[0]
        rs = check_random_state(self.random_state)
        X = np.ascontiguousarray(X, dtype=np.float64)

        self.label_binarizer_ = LabelBinarizer(neg_label=-1, pos_label=1)
        Y = self.label_binarizer_.fit_transform(y)
        self.classes_ = self.label_binarizer_.classes_.astype(np.int32)
        n_vectors = Y.shape[1]

        self.coef_ = np.zeros((n_vectors, X.shape[0]), dtype=np.float64)
        self.errors_ = np.ones((n_vectors, n_samples), dtype=np.float64)
        self.intercept_ = np.zeros(n_vectors, dtype=np.float64)

        if kcache is None:
            kernel = self._get_kernel()
            kcache = KernelCache(kernel, n_samples,
                                 self.cache_mb, 1, self.verbose)

        for i in xrange(n_vectors):
            _primal_cd_l2svm_l2r(self, self.coef_[i], self.errors_[i],
                                 X, Y[:, i],
                                 kcache,
                                 self.kernel_regularizer,
                                 self.C, self.max_outer, self.max_inner, rs,
                                 self.tol, verbose=self.verbose)

        self._post_process(X)

        return self


def C_lower_bound(X, y, kernel=None, search_size=None, random_state=None,
                  **kernel_params):
    Y = LabelBinarizer(neg_label=-1, pos_label=1).fit_transform(y)

    if kernel is None:
        den = np.max(np.abs(np.dot(Y.T, X)))
    else:
        kernel = get_kernel(kernel, **kernel_params)
        random_state = check_random_state(random_state)
        den = _C_lower_bound_kernel(X, Y, kernel, search_size, random_state)

    if den == 0.0:
        raise ValueError('Ill-posed')

    return 0.5 / den


def C_upper_bound(X, y, clf, Cmin, Cmax, n_components, epsilon, verbose=0):
    Nmax = np.inf
    clf = clone(clf)

    while Nmax - n_components > epsilon:
        Cmid = (Cmin + Cmax) / 2

        if verbose:
            print "Fit clf for C=", Cmid

        clf.set_params(C=Cmid)
        clf.fit(X, y)
        n_sv = clf.n_support_vectors()

        if verbose:
            print "#SV", clf.n_support_vectors()

        if n_sv < n_components:
            # Regularization is too strong
            Cmin = Cmid

        elif n_sv > n_components:
            # Regularization is too light
            Cmax = Cmid
            Nmax = n_sv

    if verbose:
        print "Solution: Cmax=", Cmax, "Nmax=", Nmax

    return Cmax
