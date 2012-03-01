# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
#
# Author: Mathieu Blondel
# License: BSD

import numpy as np

cimport numpy as np

cdef extern from "math.h":
   double fabs(double)

cdef extern from "float.h":
   double DBL_MAX

def _dual_cd(X,
             np.ndarray[double, ndim=1]y,
             double C,
             loss,
             int max_iter,
             rs,
             double tol,
             int precomputed_kernel,
             int verbose):
    cdef Py_ssize_t n_samples
    cdef Py_ssize_t n_features

    cdef np.ndarray[double, ndim=1, mode='c'] w

    if precomputed_kernel:
        n_samples = X.shape[0]
    else:
        n_samples, n_features = X.shape
        w = np.zeros(n_features, dtype=np.float64)

    cdef np.ndarray[double, ndim=1, mode='c'] alpha
    alpha = np.zeros(n_samples, dtype=np.float64)

    cdef np.ndarray[long, ndim=1, mode='c'] A
    A = np.arange(n_samples)
    cdef Py_ssize_t active_size = n_samples

    cdef double U
    cdef double D_ii

    if loss == "l1":
        U = C
        D_ii = 0
    elif loss == "l2":
        U = DBL_MAX
        D_ii = 1.0 / (2 * C)

    cdef np.ndarray[double, ndim=2, mode='c'] K
    cdef np.ndarray[double, ndim=1, mode='c'] Q_bar_diag
    Q_bar_diag = np.zeros(n_samples, dtype=np.float64)

    cdef int j

    if precomputed_kernel:
        K = X

        for j in xrange(n_samples):
            Q_bar_diag[j] = K[j, j]
    else:
        for j in xrange(n_samples):
            Q_bar_diag[j] = np.dot(X[j], X[j])

    Q_bar_diag += D_ii

    cdef double M
    cdef double m
    cdef int i
    cdef double y_i
    cdef double alpha_i, alpha_old
    cdef double M_bar = DBL_MAX
    cdef double m_bar = -DBL_MAX
    cdef unsigned int it = 0
    cdef int s
    cdef double G, PG
    cdef double step

    for it in xrange(max_iter):
        rs.shuffle(A[:active_size])

        M = -DBL_MAX
        m = DBL_MAX

        s = 0
        while s < active_size:
            i = A[s]
            y_i = y[i]
            alpha_i = alpha[i]

            if precomputed_kernel:
                # G = np.dot(Q_bar, alpha)[i] - 1
                G = -1
                for j in xrange(n_samples):
                    G += K[i, j] * y[i] * y[j] * alpha[j]
                G += D_ii * alpha[i]
            else:
                # G = y_i * np.dot(w, X[i]) - 1 + D_ii * alpha_i
                G = np.dot(w, X[i])
                G = y_i * G - 1 + D_ii * alpha_i

            PG = 0

            if alpha_i == 0:
                if G > M_bar:
                    active_size -= 1
                    A[s], A[active_size] = A[active_size], A[s]
                    # Jump w/o incrementing s so as to use the swapped sample.
                    continue
                elif G < 0:
                    PG = G
            elif alpha_i == C: # the paper uses U but liblinear uses C
                if G < m_bar:
                    active_size -= 1
                    A[s], A[active_size] = A[active_size], A[s]
                    continue
                elif G > 0:
                    PG = G
            else:
                PG = G

            M = max(M, PG)
            m = min(m, PG)

            if fabs(PG) > 1e-12:
                alpha_old = alpha_i

                # the paper uses U but liblinear uses C
                alpha[i] = min(max(alpha_i - G / Q_bar_diag[i], 0.0), C)

                if not precomputed_kernel:
                    step = (alpha[i] - alpha_old) * y_i
                    w += step * X[i]

            s += 1

        # end while

        if M - m <= tol:
            if active_size == n_samples:
                if verbose >= 1:
                    print "Stopped at iteration", it
                break
            else:
                active_size = n_samples
                M_bar = DBL_MAX
                m_bar = -DBL_MAX
                continue

        M_bar = M
        m_bar = m

        if M <= 0: M_bar = DBL_MAX
        if m >= 0: m_bar = -DBL_MAX

    # end for

    if precomputed_kernel:
        return alpha
    else:
        return w
