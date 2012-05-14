# Author: Mathieu Blondel
# License: BSD

import numpy as np

from sklearn.base import ClassifierMixin, clone
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_random_state
from sklearn.metrics.pairwise import pairwise_kernels

from .base import BaseLinearClassifier, BaseKernelClassifier
from .primal_cd_fast import _primal_cd_l2svm_l1r
from .primal_cd_fast import _primal_cd_l2svm_l2r
from .primal_cd_fast import _C_lower_bound_kernel
from .kernel_fast import get_kernel, KernelCache


class PrimalLinearSVC(BaseLinearClassifier, ClassifierMixin):

    def __init__(self, C=1.0, penalty="l2", max_iter=1000, tol=1e-3,
                 termination="convergence", nz_coef_upper_bound=1000,
                 warm_start=False, random_state=None,
                 callback=None, verbose=0, n_jobs=1):
        self.C = C
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
                _primal_cd_l2svm_l2r(self, self.coef_[i], self.errors_[i],
                                     X, None, Y[:, i], indices, kcache, True,
                                     self.termination, self.nz_coef_upper_bound,
                                     self.C, self.max_iter, rs, self.tol,
                                     self.callback, verbose=self.verbose)

        return self


class PrimalSVC(BaseKernelClassifier, ClassifierMixin):

    def __init__(self, C=1.0, penalty="l1", max_iter=10, tol=1e-3,
                 kernel="linear", gamma=0.1, coef0=1, degree=4,
                 Cd=1.0, warm_debiasing=False,
                 selection="permute", search_size=60,
                 termination="convergence", sv_upper_bound=1000,
                 cache_mb=500, warm_start=False, random_state=None,
                 components=None, callback=None, verbose=0, n_jobs=1):
        self.C = C
        self.penalty = penalty
        self.max_iter = max_iter
        self.tol = tol
        self.kernel = kernel
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
        self.Cd = Cd
        self.warm_debiasing = warm_debiasing
        self.selection = selection
        self.search_size = search_size
        self.termination = termination
        self.sv_upper_bound = sv_upper_bound
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

        self.label_binarizer_ = LabelBinarizer(neg_label=-1, pos_label=1)
        Y = self.label_binarizer_.fit_transform(y)
        self.classes_ = self.label_binarizer_.classes_.astype(np.int32)
        n_vectors = Y.shape[1]

        A = X
        C = self.C
        termination = self.termination

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
                                         self.termination, self.sv_upper_bound,
                                         self.C, self.max_iter, rs, self.tol,
                                         self.callback, verbose=self.verbose)

        if self.penalty == "l1l2":
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

        if self.penalty in ("l2", "l1l2"):
            for i in xrange(n_vectors):
                _primal_cd_l2svm_l2r(self, self.coef_[i], self.errors_[i],
                                     X, A, Y[:, i], indices, kcache, False,
                                     termination, self.sv_upper_bound,
                                     C, self.max_iter, rs, self.tol,
                                     self.callback, verbose=self.verbose)

        sv = np.sum(self.coef_ != 0, axis=0, dtype=bool)
        self.support_indices_ = np.arange(A.shape[0], dtype=np.int32)[sv]

        if np.sum(sv) == 0:
            # Empty model...
            self.coef_ = None
            return self

        self._post_process(A)

        return self

class PrimalKernelSVC(BaseKernelClassifier, ClassifierMixin):

    def __init__(self, C=1.0, loss="hinge", max_iter=10, tol=1e-3,
                 kernel="linear", gamma=0.1, coef0=1, degree=4,
                 selection="permute", search_size=60,
                 termination="convergence", sv_upper_bound=1000,
                 cache_mb=500, warm_start=False, random_state=None,
                 callback=None, verbose=0, n_jobs=1):
        self.C = C
        self.loss = loss
        self.max_iter = max_iter
        self.tol = tol
        self.kernel = kernel
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
        self.selection = selection
        self.search_size = search_size
        self.termination = termination
        self.sv_upper_bound = sv_upper_bound
        self.cache_mb = cache_mb
        self.warm_start = warm_start
        self.random_state = random_state
        self.callback = callback
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.support_vectors_ = None
        self.coef_ = None

    def _L(self, K, y, coef):
        value = 0.5 * np.dot(np.dot(K, coef), coef)
        losses = np.maximum(1 - y * np.dot(K, coef), 0) ** 2
        value += self.C * np.sum(losses)
        return value

    def _Lp(self, K, y, coef, j):
        value = np.dot(coef, K[j])
        losses = np.maximum(1 - y * np.dot(K, coef), 0) ** 2
        value += -2 * self.C * np.sum(y * K[j] * losses)
        return value

    def _Lpp(self, K, y, coef, j):
        value = K[j, j]
        value += 2 * self.C * np.sum(K[j] ** 2)
        return value

    def _solve_one(self, K, y, coef, j):
        sigma = 0.01
        beta = 0.5
        L0 = self._L(K, y, coef)
        d = -self._Lp(K, y, coef, j) / self._Lpp(K, y, coef, j)
        old_coef = coef[j]
        z = d

        for i in xrange(100):
            coef[j] = old_coef + z
            Li = self._L(K, y, coef)
            if Li - L0 <= -sigma * (z ** 2):
                break
            z *= beta

    def _fit_binary(self, K, y, coef, rs):
        n_samples = K.shape[0]
        indices = np.arange(n_samples)
        rs.shuffle(indices)

        for t in xrange(self.max_iter * n_samples):
            j = indices[(t-1) % n_samples]
            self._solve_one(K, y, coef, j)

        #good = y * np.dot(K, coef) > 1
        #indices = np.arange(n_samples)
        #indices = indices[~good]
        #coef[good] = 0
        #n_sv = len(indices)

        #for t in xrange(n_sv * 3):
            #j = indices[t % n_sv]
            #self._solve_one(K, y, coef, j)

    def fit(self, X, y):
        n_samples = X.shape[0]
        rs = check_random_state(self.random_state)

        self.label_binarizer_ = LabelBinarizer(neg_label=-1, pos_label=1)
        Y = self.label_binarizer_.fit_transform(y)
        self.classes_ = self.label_binarizer_.classes_.astype(np.int32)
        n_vectors = Y.shape[1]

        self.coef_ = np.zeros((n_vectors, n_samples), dtype=np.float64)
        #self.errors_ = np.ones((n_vectors, n_samples), dtype=np.float64)
        self.intercept_ = np.zeros(n_vectors, dtype=np.float64)

        K = pairwise_kernels(X, metric=self.kernel, filter_params=True,
                             **self._kernel_params())

        for i in xrange(n_vectors):
            self._fit_binary(K, Y[:, i], self.coef_[i], rs)

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


def C_upper_bound(X, y, clf, Cmin, Cmax, sv_upper_bound, epsilon, verbose=0):
    Nmax = np.inf
    clf = clone(clf)

    while Nmax - sv_upper_bound > epsilon:
        Cmid = (Cmin + Cmax) / 2

        if verbose:
            print "Fit clf for C=", Cmid

        clf.set_params(C=Cmid)
        clf.fit(X, y)
        n_sv = clf.n_support_vectors()

        if verbose:
            print "#SV", clf.n_support_vectors()

        if n_sv < sv_upper_bound:
            # Regularization is too strong
            Cmin = Cmid

        elif n_sv > sv_upper_bound:
            # Regularization is too light
            Cmax = Cmid
            Nmax = n_sv

    if verbose:
        print "Solution: Cmax=", Cmax, "Nmax=", Nmax

    return Cmax
