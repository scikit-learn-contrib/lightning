# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
#
# Author: Mathieu Blondel
# License: BSD

import sys

import numpy as np
cimport numpy as np

from lightning.random.random_fast cimport RandomState
from lightning.dataset_fast cimport RowDataset

cdef extern from "math.h":
   double fabs(double)

cdef extern from "float.h":
   double DBL_MAX


def _dual_cd(self,
             np.ndarray[double, ndim=1, mode='c'] w,
             np.ndarray[double, ndim=1, mode='c'] alpha,
             RowDataset X,
             np.ndarray[double, ndim=1]y,
             int permute,
             double C,
             int loss,
             int max_iter,
             RandomState rs,
             double tol,
             int shrinking,
             callback,
             int verbose):
    cdef Py_ssize_t n_samples = X.get_n_samples()
    cdef Py_ssize_t n_features = X.get_n_features()

    # Initialization.
    cdef double* col
    cdef double M
    cdef double m
    cdef int i, j, jj
    cdef double y_i
    cdef double alpha_i, alpha_old
    cdef double M_bar = DBL_MAX
    cdef double m_bar = -DBL_MAX
    cdef unsigned int t = 0
    cdef int s
    cdef double G, PG
    cdef double step
    cdef int r
    cdef int has_callback = callback is not None
    cdef int stop = 0
    cdef double U
    cdef double D_ii

    # Loss-dependent values.
    if loss == 1: # hinge
        U = C
        D_ii = 0
    elif loss == 2: # squared_hinge
        U = DBL_MAX
        D_ii = 1.0 / (2 * C)

    # Active set.
    cdef np.ndarray[int, ndim=1, mode='c'] A
    A = np.arange(n_samples, dtype=np.int32)
    cdef Py_ssize_t active_size = n_samples

    # Data pointers.
    cdef double* data
    cdef int* indices
    cdef int n_nz

    # Diagonal values of the Q matrix (squared norms).
    cdef np.ndarray[double, ndim=1, mode='c'] Q_bar_diag
    Q_bar_diag = np.zeros(n_samples, dtype=np.float64)
    for i in xrange(n_samples):
        X.get_row_ptr(i, &indices, &data, &n_nz)
        for jj in xrange(n_nz):
            Q_bar_diag[i] += data[jj] * data[jj]
    Q_bar_diag += D_ii


    for t in xrange(max_iter):
        if verbose >= 1:
            print "\nIteration", t

        if permute:
            rs.shuffle(A[:active_size])

        M = -DBL_MAX
        m = DBL_MAX

        s = 0
        while s < active_size:
            i = A[s]

            y_i = y[i]
            alpha_i = fabs(alpha[i])

            # Compute ith element of the gradient.
            X.get_row_ptr(i, &indices, &data, &n_nz)

            # G = y_i * np.dot(w, X[i]) - 1 + D_ii * alpha_i
            G = 0
            for jj in xrange(n_nz):
                j = indices[jj]
                G += w[j] * data[jj]
            G = y_i * G - 1 + D_ii * alpha_i

            PG = 0

            # Shrinking.
            if alpha_i == 0:
                if G < 0:
                    PG = G
                elif G > M_bar and shrinking:
                    active_size -= 1
                    A[s], A[active_size] = A[active_size], A[s]
                    # Jump w/o incrementing s so as to use the swapped sample.
                    continue
            elif alpha_i == U:
                if G > 0:
                    PG = G
                elif G < m_bar and shrinking:
                    active_size -= 1
                    A[s], A[active_size] = A[active_size], A[s]
                    continue
            else:
                PG = G

            M = max(M, PG)
            m = min(m, PG)

            if fabs(PG) > 1e-12:
                alpha_old = alpha_i

                # Closed-form solution of the one-variable subproblem.
                alpha_i = min(max(alpha_i - G / Q_bar_diag[i], 0), U)
                alpha[i] = alpha_i * y_i

                # Update the primal coefficients.
                step = (alpha_i - alpha_old) * y_i
                for jj in xrange(n_nz):
                    j = indices[jj]
                    w[j] += step * data[jj]

            # Callback
            if has_callback and s % 100 == 0:
                ret = callback(self)
                if ret is not None:
                    stop = 1
                    break

            # Output progress.
            if verbose >= 1 and s % 100 == 0:
                sys.stdout.write(".")
                sys.stdout.flush()

            s += 1

        # end while

        if stop:
            break

        # Convergence check.
        if M - m <= tol:
            if active_size == n_samples:
                if verbose >= 1:
                    print "\nConverged at iteration", t
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

    if verbose >= 1:
        print

    return w, alpha
