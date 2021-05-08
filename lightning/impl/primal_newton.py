"""
=============================
Kernel SVM by Newton's method
=============================

This module provides a kernel SVM solver using Newton's method.

"""
# Author: Mathieu Blondel
# License: BSD

import numpy as np

from scipy.sparse.linalg import cg
from scipy.linalg import solve

from sklearn.utils import safe_mask
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_random_state
from sklearn.metrics.pairwise import pairwise_kernels

from .base import BaseClassifier


class KernelSVC(BaseClassifier):
    r"""Estimator for learning kernel SVMs by Newton's method.

    Parameters
    ----------

    alpha : float
        Weight of the penalty term.

    solver : str, 'cg', 'dense'

    max_iter : int
        Maximum number of iterations to perform.

    tol : float
        Tolerance of the stopping criterion.

    kernel: "linear" | "poly" | "rbf" | "sigmoid" | "cosine" | "precomputed"
        Kernel to use.  Default: "linear"

    degree : int, default=3
        Degree for poly, rbf and sigmoid kernels. Ignored by other kernels.

    gamma : float, optional
        Kernel coefficient for rbf and poly kernels. Default: 1/n_features.
        Ignored by other kernels.

    coef0 : float, optional
        Independent term in poly and sigmoid kernels.
        Ignored by other kernels.

    random_state : RandomState or int
        The seed of the pseudo random number generator to use.

    verbose : int
        Verbosity level.

    n_jobs : int
        Number of jobs to use to compute the kernel matrix.

    Example
    -------

    >>> from sklearn.datasets import make_classification
    >>> from lightning.classification import KernelSVC
    >>> X, y = make_classification()
    >>> clf = KernelSVC().fit(X, y)
    >>> accuracy = clf.score(X, y)
    """

    def __init__(self, alpha=1.0, solver="cg",
                 max_iter=50, tol=1e-3,
                 kernel="linear", gamma=0.1, coef0=1, degree=4,
                 random_state=None, verbose=0, n_jobs=1):
        self.alpha = alpha
        self.solver = solver
        self.max_iter = max_iter
        self.tol = tol
        self.kernel = kernel
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
        self.random_state = random_state
        self.verbose = verbose
        self.n_jobs = n_jobs

    def _kernel_params(self):
        return {"gamma": self.gamma,
                "degree": self.degree,
                "coef0": self.coef0}

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

        for t in range(1, self.max_iter + 1):
            if self.verbose:
                print("Iteration", t, "#SV=", np.sum(sv))

            K_sv = K[sv][:, sv]
            I = np.diag(self.alpha * np.ones(K_sv.shape[0]))

            coef_sv = self._solve(K_sv + I, y[sv])

            coef *= 0
            coef[sv] = coef_sv
            pred = np.dot(K, coef)
            errors = 1 - y * pred
            last_sv = sv
            sv = errors > 0

            if np.array_equal(last_sv, sv):
                if self.verbose:
                    print("Converged at iteration", t)
                break

        return coef

    def _post_process(self, X):
        # We can't know the support vectors when using precomputed kernels.
        if self.kernel != "precomputed":
            sv = np.sum(self.coef_ != 0, axis=0, dtype=bool)
            if np.sum(sv) > 0:
                self.coef_ = np.ascontiguousarray(self.coef_[:, sv])
                mask = safe_mask(X, sv)
                self.support_vectors_ = np.ascontiguousarray(X[mask])
                self.support_indices_ = np.arange(X.shape[0],
                                                  dtype=np.int32)[sv]
                self.n_samples_ = X.shape[0]

            if self.verbose >= 1:
                print("Number of support vectors:", np.sum(sv))

    def fit(self, X, y):
        """Fit model according to X and y.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : classifier
            Returns self.
        """
        n_samples, n_features = X.shape
        rs = check_random_state(self.random_state)

        self.label_binarizer_ = LabelBinarizer(neg_label=-1, pos_label=1)
        Y = self.label_binarizer_.fit_transform(y)
        self.classes_ = self.label_binarizer_.classes_.astype(np.int32)
        n_vectors = Y.shape[1]

        if self.verbose:
            print("Pre-computing kernel matrix...")

        K = pairwise_kernels(X, filter_params=True, n_jobs=self.n_jobs,
                             metric=self.kernel, **self._kernel_params())

        coef = [self._fit_binary(K, Y[:, i], rs) for i in range(n_vectors)]
        self.coef_ = np.array(coef)
        self.intercept_ = np.zeros(n_vectors, dtype=np.float64)

        self._post_process(X)

        return self

    def decision_function(self, X):
        """
        Return the decision function for test vectors X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        P : array, shape = [n_classes, n_samples]
            Decision function for X
        """
        K = pairwise_kernels(X, self.support_vectors_, filter_params=True,
                             n_jobs=self.n_jobs, metric=self.kernel,
                             **self._kernel_params())
        return np.dot(K, self.coef_.T)
