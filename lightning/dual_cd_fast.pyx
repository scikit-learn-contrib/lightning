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


cdef double _sparse_dot(double* data1,
                        int* indices1,
                        int n_nz1,
                        double* data2,
                        int* indices2,
                        int n_nz2):

    cdef int i = 0
    cdef int j = 0
    cdef double dot = 0

    while i < n_nz1 and j < n_nz2:
        if indices1[i] == indices2[j]:
            dot += data1[i] * data2[j]
            i += 1
            j += 1
        else:
            if indices1[i] > indices2[j]:
                j += 1
            else:
                i += 1

    return dot


# For test purposes only.
def sparse_dot(RowDataset X, i, j):
    cdef double* data1
    cdef int* indices1
    cdef int n_nz1
    X.get_row_ptr(i, &indices1, &data1, &n_nz1)

    cdef double* data2
    cdef int* indices2
    cdef int n_nz2
    X.get_row_ptr(j, &indices2, &data2, &n_nz2)

    return _sparse_dot(data1, indices1, n_nz1,
                       data2, indices2, n_nz2)


cdef _sqnorms(RowDataset X,
              np.ndarray[double, ndim=1, mode='c'] sqnorms):

    cdef Py_ssize_t n_samples = X.get_n_samples()
    cdef int i, j
    cdef double dot

    # Data pointers.
    cdef double* data
    cdef int* indices
    cdef int n_nz

    for i in xrange(n_samples):
        X.get_row_ptr(i, &indices, &data, &n_nz)
        dot = 0
        for jj in xrange(n_nz):
            dot += data[jj] * data[jj]
        sqnorms[i] = dot


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
             int n_calls,
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
    _sqnorms(X, Q_bar_diag)
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

            # Retrieve row.
            X.get_row_ptr(i, &indices, &data, &n_nz)

            # Compute ith element of the gradient.
            # G = y_i * np.dot(w, X[i]) - 1 + D_ii * alpha_i
            G = 0
            for jj in xrange(n_nz):
                j = indices[jj]
                G += w[j] * data[jj]
            G = y_i * G - 1 + D_ii * alpha_i

            # Projected gradient and shrinking.
            PG = 0
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
            if has_callback and s % n_calls == 0:
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
                # When shrinking is enabled, we need to do one more outer
                # iteration on the entire optimization problem.
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


def _dual_cd_auc(self,
                 np.ndarray[double, ndim=1, mode='c'] w,
                 RowDataset X,
                 np.ndarray[double, ndim=1]y,
                 double C,
                 int loss,
                 int max_iter,
                 RandomState rs,
                 int verbose):

    cdef Py_ssize_t n_samples = X.get_n_samples()
    cdef Py_ssize_t n_features = X.get_n_features()

    # Initialization.
    cdef int r, s, p, n, j, jj, t, tt, k
    cdef double U
    cdef double D_ii
    cdef double alpha_k, alpha_old, denom
    cdef double G, PG

    # Loss-dependent values.
    if loss == 1: # hinge
        U = C
        D_ii = 0
    elif loss == 2: # squared_hinge
        U = DBL_MAX
        D_ii = 1.0 / (2 * C)

    # Data pointers.
    cdef double* data1
    cdef int* indices1
    cdef int n_nz1

    cdef double* data2
    cdef int* indices2
    cdef int n_nz2

    # Diagonal values of the Q matrix (squared norms).
    cdef np.ndarray[double, ndim=1, mode='c'] sqnorms
    sqnorms = np.zeros(n_samples, dtype=np.float64)
    _sqnorms(X, sqnorms)
    sqnorms += D_ii

    # Index of positive and negative examples.
    cdef np.ndarray[int, ndim=1, mode='c'] pos
    pos = np.zeros(n_samples, dtype=np.int32)
    cdef np.ndarray[int, ndim=1, mode='c'] neg
    neg = np.zeros(n_samples, dtype=np.int32)

    cdef int n_pos = 0
    cdef int n_neg = 0

    for i in xrange(n_samples):
        if y[i] == 1:
            pos[n_pos] = i
            n_pos += 1
        else:
            neg[n_neg] = i
            n_neg += 1

    # Dual coefs
    cdef np.ndarray[double, ndim=1, mode='c'] alpha
    alpha = np.zeros(n_pos * n_neg, dtype=np.float64)

    # Learning
    for t in xrange(max_iter):
        if verbose >= 1:
            print "\nIteration", t

        for tt in xrange(n_samples):
            r = rs.randint(n_pos - 1)
            s = rs.randint(n_neg - 1)
            p = pos[r]
            n = neg[s]
            k = r * n_neg + s

            # Retrieve positive sample.
            X.get_row_ptr(p, &indices1, &data1, &n_nz1)

            # Retrieve negative sample.
            X.get_row_ptr(n, &indices2, &data2, &n_nz2)

            alpha_k = fabs(alpha[k])

            # Gradient
            G = 0
            for jj in xrange(n_nz1):
                j = indices1[jj]
                G += w[j] * data1[jj]

            for jj in xrange(n_nz2):
                j = indices2[jj]
                G -= w[j] * data2[jj]

            G = G - 1 + D_ii * alpha_k

            # Projected gradient
            PG = 0
            if alpha_k == 0:
                if G < 0:
                    PG = G
            elif alpha_k == U:
                if G > 0:
                    PG = G
            else:
                PG = G

            # Update alpha.
            if fabs(PG) > 1e-12:
                denom = sqnorms[p] + sqnorms[n]
                denom -= 2 * _sparse_dot(data1, indices1, n_nz1,
                                     data2, indices2, n_nz2)

                alpha_old = alpha_k
                alpha_k = min(max(alpha_k - G / denom, 0), U)
                alpha[k] = alpha_k

                # Update w.
                step = (alpha_k - alpha_old)

                for jj in xrange(n_nz1):
                    j = indices1[jj]
                    w[j] += step * data1[jj]

                for jj in xrange(n_nz2):
                    j = indices2[jj]
                    w[j] -= step * data2[jj]

    return w, alpha
