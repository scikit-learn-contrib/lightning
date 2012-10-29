# Author: Mathieu Blondel
# License: BSD

import numpy as np

from scipy.sparse.linalg import cg
from scipy.linalg import solve

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_random_state
from sklearn.metrics.pairwise import pairwise_kernels

from .base import BaseKernelClassifier

class PrimalNewton(BaseKernelClassifier, ClassifierMixin):

    def __init__(self, lmbda=1.0, solver="cg",
                 max_iter=50, tol=1e-3, n_components=None,
                 kernel="linear", gamma=0.1, coef0=1, degree=4,
                 random_state=0, verbose=0, n_jobs=1):
        self.lmbda = lmbda
        self.solver = solver
        self.max_iter = max_iter
        self.tol = tol
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
        self.random_state = random_state
        self.verbose = verbose
        self.n_jobs = n_jobs

    def _solve(self, A, b):
        if self.solver == "cg":
            x, info = cg(A, b, tol=self.tol)
        elif self.solver == "dense":
            x = solve(A, b, sym_pos=True)
        return x

    def _fit_binary(self, K, y, rs):
        n_samples = K.shape[0]
        coef = np.zeros(n_samples)
        if n_samples < 1000:
            sv = np.ones(n_samples, dtype=bool)
        else:
            sv = np.zeros(n_samples, dtype=bool)
            sv[:1000] = True
            rs.shuffle(sv)

        for t in xrange(1, self.max_iter + 1):
            if self.verbose:
                print "Iteration", t, "#SV=", np.sum(sv)

            K_sv = K[sv][:, sv]
            I = np.diag(self.lmbda * np.ones(K_sv.shape[0]))

            coef_sv = self._solve(K_sv + I, y[sv])

            coef *= 0
            coef[sv] = coef_sv
            pred = np.dot(K, coef)
            errors = 1 - y * pred
            last_sv = sv
            sv = errors > 0

            if np.array_equal(last_sv, sv):
                if self.verbose:
                    print "Converged at iteration", t
                break

        return coef

    def _fit_binary_inc(self, K, y, rs):
        n_samples = K.shape[0]
        coef = np.zeros(n_samples)

        is_sel = {rs.randint(n_samples) : 1}
        for t in xrange(1, self.n_components + 1):
            if self.verbose:
                print "#SV", t

            J = np.array(is_sel.keys())
            K_J = K[J]
            coef_J = coef[J]
            pred = np.dot(K_J.T, coef_J)
            errors = 1 - y * pred
            I = errors > 0
            K_JI = K_J[:, I]
            K_JJ = K_J[:, J]
            P = self.lmbda * K_JJ + np.dot(K_JI, K_JI.T)
            g = np.dot(self.lmbda * K_JJ, coef_J)
            g -= np.dot(K_JI, y[I] - pred[I])

            sol = self._solve(P, g)
            coef[J] = coef_J - sol

            ind = errors.argsort()[::-1]
            for i in ind:
                if not i in is_sel:
                    is_sel[i] = 1
                    break

        return coef

    def fit(self, X, y):
        n_samples, n_features = X.shape
        rs = check_random_state(self.random_state)

        self.label_binarizer_ = LabelBinarizer(neg_label=-1, pos_label=1)
        Y = self.label_binarizer_.fit_transform(y).astype(np.float64)
        self.classes_ = self.label_binarizer_.classes_.astype(np.int32)
        n_vectors = Y.shape[1]

        if self.verbose:
            print "Pre-computing kernel matrix..."

        K = pairwise_kernels(X, filter_params=True, n_jobs=self.n_jobs,
                             metric=self.kernel, **self._kernel_params())


        func = self._fit_binary if self.n_components is None \
                                else self._fit_binary_inc

        coef = [func(K, Y[:, i], rs) for i in xrange(n_vectors)]
        self.coef_ = np.array(coef)
        self.intercept_ = np.zeros(n_vectors, dtype=np.float64)

        self._post_process(X)

        return self
