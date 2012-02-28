# Author: Mathieu Blondel
# License: BSD

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_random_state


def _dual_cd(X, y, C, loss, max_iter, rs, tol=1e-4, precomputed=False,
             verbose=False):
    if precomputed:
        n_samples = X.shape[0]
    else:
        n_samples, n_features = X.shape
        w = np.zeros(n_features, dtype=np.float64)

    alpha = np.zeros(n_samples, dtype=np.float64)
    A = np.arange(n_samples)
    active_size = n_samples

    if loss == "l1":
        U = C
        D_ii = 0
    elif loss == "l2":
        U = np.inf
        D_ii = 1.0 / (2 * C)

    if precomputed:
        Q_bar = X * np.outer(y, y)
        Q_bar += np.eye(n_samples) * D_ii

    M_bar = np.inf
    m_bar = -np.inf

    for it in xrange(max_iter):
        rs.shuffle(A[:active_size])

        M = -np.inf
        m = np.inf

        s = -1
        #for (s=0; s<active_size; s++)
        while s < active_size-1:
            s += 1
            i = A[s]
            y_i = y[i]
            alpha_i = alpha[i]

            if precomputed:
                # Need to be optimized in cython
                #G = -1
                #for j in xrange(n_samples):
                    #G += Q_bar[i, j] * alpha[j]
                G = np.dot(Q_bar, alpha)[i] - 1
            else:
                G = y_i * np.dot(w, X[i]) - 1 + D_ii * alpha_i

            PG = 0

            if alpha_i == 0:
                if G > M_bar:
                    active_size -= 1
                    A[s], A[active_size] = A[active_size], A[s]
                    s -= 1
                    continue
                elif G < 0:
                    PG = G
            elif alpha_i == U:
                if G < m_bar:
                    active_size -= 1
                    A[s], A[active_size] = A[active_size], A[s]
                    s -= 1
                    continue
                elif G > 0:
                    PG = G
            else:
                PG = G

            M = max(M, PG)
            m = min(m, PG)

            if np.abs(PG) > 1e-12:
               alpha_old = alpha_i

               if precomputed:
                   Q_bar_ii = Q_bar[i, i]
               else:
                # FIXME: can be pre-computed
                   Q_bar_ii = np.dot(X[i], X[i]) + D_ii

               alpha[i] = min(max(alpha_i - G / Q_bar_ii, 0.0), U)

               if not precomputed:
                   w += (alpha[i] - alpha_old) * y_i * X[i]

        if M - m <= tol:
            if active_size == n_samples:
                if verbose >= 1:
                    print "Stopped at iteration", it
                break
            else:
                active_size = n_samples
                M_bar = np.inf
                m_bar = -np.inf
                continue

        M_bar = M
        m_bar = m

        if M <= 0: M_bar = np.inf
        if m >= 0: m_bar = -np.inf

    if precomputed:
        return alpha
    else:
        return w


class DualLinearSVC(BaseEstimator):

    def __init__(self, C=1.0, loss="l1", max_iter=1000, random_state=None,
                 verbose=0):
        self.C = C
        self.loss = loss
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        rs = check_random_state(self.random_state)
        self.label_binarizer_ = LabelBinarizer(neg_label=-1, pos_label=1)
        Y = self.label_binarizer_.fit_transform(y)
        W = [_dual_cd(X, Y[:, i],
                      self.C, self.loss, self.max_iter, rs,
                      precomputed=False, verbose=self.verbose) \
                for i in range(Y.shape[1])]
        self.coef_ = np.array(W)

    def decision_function(self, X):
        return np.dot(X, self.coef_.T)

    def predict(self, X):
        pred = self.decision_function(X)
        return self.label_binarizer_.inverse_transform(pred, threshold=0)


class DualSVC(BaseEstimator):

    def __init__(self, C=1.0, loss="l1", max_iter=1000, random_state=None,
                 verbose=0):
        self.C = C
        self.loss = loss
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        rs = check_random_state(self.random_state)
        self.label_binarizer_ = LabelBinarizer(neg_label=-1, pos_label=1)
        Y = self.label_binarizer_.fit_transform(y)
        K = np.dot(X, X.T)
        Alpha = [_dual_cd(K, Y[:, i],
                          self.C, self.loss, self.max_iter, rs,
                          precomputed=True, verbose=self.verbose)
                    for i in range(Y.shape[1])]
        self.dual_coef_ = np.array(Alpha) * Y.T
        # FIXME: can trim the model
        self.X_train_ = X

    def decision_function(self, X):
        K = np.dot(X, self.X_train_.T)
        return np.dot(K, self.dual_coef_.T)

    def predict(self, X):
        pred = self.decision_function(X)
        return self.label_binarizer_.inverse_transform(pred, threshold=0)
