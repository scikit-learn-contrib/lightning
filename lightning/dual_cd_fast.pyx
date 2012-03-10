# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
#
# Author: Mathieu Blondel
# License: BSD

from cython.operator cimport dereference as deref
from cython.operator cimport preincrement as inc
from cython.operator cimport predecrement as dec

from libcpp.list cimport list as list
from libcpp.vector cimport vector

import numpy as np
cimport numpy as np

from lightning.kernel_fast cimport Kernel
from lightning.select_fast cimport get_select_method, select_sv

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
             selection,
             int search_size,
             termination,
             int sv_upper_bound,
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
    cdef unsigned int t = 0
    cdef int s, start = 0
    cdef double G, PG
    cdef double step
    cdef long r

    cdef list[long] support_set
    cdef list[long].iterator it

    cdef vector[list[long].iterator] support_it
    support_it.resize(n_samples)

    cdef np.ndarray[long, ndim=1, mode='c'] support_vectors
    support_vectors = np.zeros(n_samples, dtype=np.int64)

    cdef int select_method = get_select_method(selection)
    cdef int check_n_sv = termination == "n_sv"
    cdef int check_convergence = termination == "convergence"
    cdef int stop = 0

    # FIXME: would be better to store the support indices in the class
    for i in xrange(n_samples):
        if alpha[i] != 0:
            support_set.push_back(i)
            support_vectors[i] = 1
            it = support_set.end()
            dec(it)
            support_it[i] = it

    for t in xrange(max_iter):
        rs.shuffle(A[:active_size])

        M = -DBL_MAX
        m = DBL_MAX

        s = 0
        start = 0
        while s < active_size:
            i = select_sv(A, start, search_size, active_size, select_method,
                          alpha, 0, X, y, kernel,
                          support_set, support_vectors)

            y_i = y[i]
            alpha_i = alpha[i]

            # Compute ith element of the gradient.
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
                it = support_set.begin()
                while it != support_set.end():
                    j = deref(it)
                    G += col_data[j] * y[i] * y[j] * alpha[j]
                    inc(it)
                G += D_ii * alpha[i]

            PG = 0

            # Shrinking.
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

                # Closed-form solution of the one-variable subproblem.
                alpha[i] = min(max(alpha_i - G / Q_bar_diag[i], 0), U)

                # Update support set.
                if alpha[i] != 0:
                    if support_vectors[i] == 0:
                        support_set.push_back(i)
                        it = support_set.end()
                        dec(it)
                        support_it[i] = it
                        support_vectors[i] = 1
                elif alpha[i] == 0:
                    if support_vectors[i] == 1:
                        it = support_it[i]
                        support_set.erase(it)
                        support_vectors[i] = 0

                if linear_kernel:
                    step = (alpha[i] - alpha_old) * y_i
                    w += step * X[i]

            # Exit if necessary.
            if support_set.size() == n_samples or \
               (check_n_sv and support_set.size() >= sv_upper_bound):
                stop = 1
                break

            # Update position and reshuffle if needed.
            if select_method: # others than permute
                start += search_size

                if start + search_size > n_samples - 1:
                    rs.shuffle(A[:active_size])
                    start = 0
            else:
                start += 1

            s += 1

        # end while

        if stop:
            break

        # Convergence check.
        if check_convergence and M - m <= tol:
            if active_size == n_samples:
                if verbose >= 1:
                    print "Stopped at iteration", t
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
