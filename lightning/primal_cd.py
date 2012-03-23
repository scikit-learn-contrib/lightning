# Author: Mathieu Blondel
# License: BSD

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_random_state, safe_mask
from sklearn.metrics.pairwise import pairwise_kernels

from .primal_cd_fast import _primal_cd_l2svm_l1r
from .primal_cd_fast import _primal_cd_l2svm_l2r
from .primal_cd_fast import _C_lower_bound_kernel
from .kernel_fast import get_kernel, KernelCache
from .predict_fast import predict_alpha, decision_function_alpha

class PrimalLinearSVC(BaseEstimator, ClassifierMixin):

    def __init__(self, C=1.0, penalty="l2", max_iter=1000, tol=1e-3,
                 termination="convergence", nz_coef_upper_bound=1000,
                 warm_start=False, random_state=None, verbose=0, n_jobs=1):
        self.C = C
        self.penalty = penalty
        self.max_iter = max_iter
        self.tol = tol
        self.termination = termination
        self.nz_coef_upper_bound = nz_coef_upper_bound
        self.warm_start = warm_start
        self.random_state = random_state
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
        kcache = KernelCache(kernel, n_samples, 0, self.verbose)

        if not self.warm_start or self.coef_ is None:
            self.coef_ = np.zeros((n_vectors, n_features), dtype=np.float64)
            self.errors_ = np.ones((n_vectors, n_samples), dtype=np.float64)

        for i in xrange(n_vectors):
            if self.penalty == "l1":
                _primal_cd_l2svm_l1r(self.coef_[i], self.errors_[i],
                                     X, Y[:, i], kcache, True,
                                     "permute", 60,
                                     self.termination, self.nz_coef_upper_bound,
                                     self.C, self.max_iter, rs, self.tol,
                                     verbose=self.verbose)
            else:
                _primal_cd_l2svm_l2r(self.coef_[i], self.errors_[i],
                                     X, None, Y[:, i], kcache, True,
                                     self.termination, self.nz_coef_upper_bound,
                                     self.C, self.max_iter, rs, self.tol,
                                     verbose=self.verbose)

        return self

    def decision_function(self, X):
        return np.dot(X, self.coef_.T)

    def predict(self, X):
        pred = self.decision_function(X)
        return self.label_binarizer_.inverse_transform(pred, threshold=0)


class PrimalSVC(BaseEstimator, ClassifierMixin):

    def __init__(self, C=1.0, penalty="l1", max_iter=10, tol=1e-3,
                 kernel="linear", gamma=0.1, coef0=1, degree=4,
                 Cd=1.0, warm_debiasing=False,
                 selection="permute", search_size=60,
                 termination="convergence", sv_upper_bound=1000,
                 cache_mb=500, warm_start=False, random_state=None,
                 verbose=0, n_jobs=1):
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
        self.warm_start = warm_start
        self.random_state = random_state
        self.cache_mb = cache_mb
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.support_vectors_ = None
        self.coef_ = None

    def _kernel_params(self):
        return {"gamma" : self.gamma,
                "degree" : self.degree,
                "coef0" : self.coef0}

    def _get_kernel(self):
        return get_kernel(self.kernel, **self._kernel_params())

    def fit(self, X, y):
        n_samples = X.shape[0]
        rs = check_random_state(self.random_state)

        self.label_binarizer_ = LabelBinarizer(neg_label=-1, pos_label=1)
        Y = self.label_binarizer_.fit_transform(y)
        n_vectors = Y.shape[1]

        if not self.warm_start or self.coef_ is None:
            self.coef_ = np.zeros((n_vectors, n_samples), dtype=np.float64)
            self.errors_ = np.ones((n_vectors, n_samples), dtype=np.float64)

        kernel = self._get_kernel()
        kcache = KernelCache(kernel, n_samples, self.cache_mb * 1024 * 1024,
                             self.verbose)

        if self.penalty in ("l1", "l1l2"):
            for i in xrange(n_vectors):
                    _primal_cd_l2svm_l1r(self.coef_[i], self.errors_[i],
                                         X, Y[:, i], kcache, False,
                                         self.selection, self.search_size,
                                         self.termination, self.sv_upper_bound,
                                         self.C, self.max_iter, rs, self.tol,
                                         verbose=self.verbose)

        A = X
        C = self.C
        termination = self.termination

        if self.penalty == "l1l2":
            sv = np.sum(self.coef_ != 0, axis=0, dtype=bool)
            A = X[sv]
            kcache = KernelCache(kernel, n_samples,
                                 self.cache_mb * 1024 * 1024, self.verbose)
            if self.warm_debiasing:
                self.coef_ = np.ascontiguousarray(self.coef_[:, sv])
            else:
                self.coef_ = np.zeros((n_vectors, A.shape[0]), dtype=np.float64)
                self.errors_ = np.ones((n_vectors, n_samples), dtype=np.float64)
            C = self.Cd
            termination = "convergence"

        if self.penalty in ("l2", "l1l2"):
            for i in xrange(n_vectors):
                _primal_cd_l2svm_l2r(self.coef_[i], self.errors_[i],
                                     X, A, Y[:, i], kcache, False,
                                     termination, self.sv_upper_bound,
                                     C, self.max_iter, rs, self.tol,
                                     verbose=self.verbose)

        if self.penalty == "l1l2" and self.warm_start:
            # Need to restore the original size of coef_.
            coef = np.zeros((n_vectors, n_samples), dtype=np.float64)
            coef[:, sv] = self.coef_
            self.coef_ = coef
            A = X

        sv = np.sum(self.coef_ != 0, axis=0, dtype=bool)

        if np.sum(sv) == 0:
            # Empty model...
            self.coef_ = None
            return self

        # We can't know the support vectors when using precomputed kernels.
        if self.kernel != "precomputed":
            self.support_vectors_ = A

        # Cannot trim the non-zero weights if warm start is used...
        if not self.warm_start:
            self.coef_ = np.ascontiguousarray(self.coef_[:, sv])
            mask = safe_mask(X, sv)
            self.support_vectors_ = A[mask]

        self.classes_ = self.label_binarizer_.classes_.astype(np.int32)

        if self.verbose >= 1:
            print "Number of support vectors:", np.sum(sv)

        return self

    def n_support_vectors(self):
        return 0 if self.coef_ is None else np.sum(self.coef_ != 0)

    def decision_function(self, X):
        out = np.zeros((X.shape[0], self.coef_.shape[0]), dtype=np.float64)
        if self.coef_ is not None:
            sv = self.support_vectors_ if self.kernel != "precomputed" else X
            decision_function_alpha(X, sv, self.coef_, self._get_kernel(), out)
        return out

    def predict(self, X):
        out = np.zeros(X.shape[0], dtype=np.float64)
        if self.coef_ is not None:
            sv = self.support_vectors_ if self.kernel != "precomputed" else X
            predict_alpha(X, sv, self.coef_, self.classes_,
                          self._get_kernel(), out)
        return out


def C_lower_bound(X, y, kernel=None, search_size=None, random_state=None,
                  **kernel_params):
    classes = np.unique(y)
    n_classes = np.size(classes)

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
