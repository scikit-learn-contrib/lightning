# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
#
# Author: Mathieu Blondel
# License: BSD

import numpy as np
cimport numpy as np

from lightning.kernel_fast cimport Kernel

cdef extern from "math.h":
   double fabs(double)

cdef extern from "float.h":
   double DBL_MAX

def _dual_cd(np.ndarray[double, ndim=1, mode='c'] w,
             np.ndarray[double, ndim=1, mode='c'] alpha,
             np.ndarray[double, ndim=2, mode='c'] X,
             np.ndarray[double, ndim=1]y,
             Kernel kernel,
             int linear_kernel,
             double C,
             loss,
             int max_iter,
             rs,
             double tol,
             int shrinking,
             int verbose):
    cdef Py_ssize_t n_samples = X.shape[0]
    cdef Py_ssize_t n_features = X.shape[1]

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

    cdef double* col_data
    cdef np.ndarray[double, ndim=1, mode='c'] col
    col = np.zeros(n_samples, dtype=np.float64)
    col_data = <double*>col.data

    cdef np.ndarray[double, ndim=1, mode='c'] Q_bar_diag
    Q_bar_diag = np.zeros(n_samples, dtype=np.float64)

    kernel.compute_diag_ptr(X, <double*>Q_bar_diag.data)
    Q_bar_diag += D_ii

    cdef double M
    cdef double m
    cdef int i, j
    cdef double y_i
    cdef double alpha_i, alpha_old
    cdef double M_bar = DBL_MAX
    cdef double m_bar = -DBL_MAX
    cdef unsigned int it = 0
    cdef int s
    cdef double G, PG
    cdef double step

    for it in xrange(max_iter):
        # FIXME: Could select instances greedily via randomized search instead
        #        of randomly
        rs.shuffle(A[:active_size])

        M = -DBL_MAX
        m = DBL_MAX

        s = 0
        while s < active_size:
            i = A[s]
            y_i = y[i]
            alpha_i = alpha[i]

            if linear_kernel:
                # G = y_i * np.dot(w, X[i]) - 1 + D_ii * alpha_i
                G = 0
                for j in xrange(n_features):
                    G += w[j] * X[i, j]
                G = y_i * G - 1 + D_ii * alpha_i
            else:
                # G = np.dot(Q_bar, alpha)[i] - 1
                G = -1
                # FIXME: retrieve sv only and iterate over non-zero alpha[j]
                kernel.compute_column_ptr(X, X, i, col_data)
                for j in xrange(n_samples):
                    G += col_data[j] * y[i] * y[j] * alpha[j]
                G += D_ii * alpha[i]

            PG = 0

            if alpha_i == 0:
                if G < 0 or not shrinking:
                    PG = G
                elif G > M_bar:
                    active_size -= 1
                    A[s], A[active_size] = A[active_size], A[s]
                    # Jump w/o incrementing s so as to use the swapped sample.
                    continue
            elif alpha_i == U:
                if G > 0 or not shrinking:
                    PG = G
                elif G < m_bar:
                    active_size -= 1
                    A[s], A[active_size] = A[active_size], A[s]
                    continue
            else:
                PG = G

            M = max(M, PG)
            m = min(m, PG)

            if fabs(PG) > 1e-12:
                alpha_old = alpha_i

                alpha[i] = min(max(alpha_i - G / Q_bar_diag[i], 0), U)

                if linear_kernel:
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

    if linear_kernel:
        return w
    else:
        return alpha
