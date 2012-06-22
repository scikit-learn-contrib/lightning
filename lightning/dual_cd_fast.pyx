# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
#
# Author: Mathieu Blondel
# License: BSD

import sys

from cython.operator cimport dereference as deref
from cython.operator cimport preincrement as inc
from cython.operator cimport predecrement as dec

from libcpp.list cimport list
from libcpp.vector cimport vector

import numpy as np
cimport numpy as np

from lightning.kernel_fast cimport KernelCache
from lightning.select_fast cimport get_select_method, select_sv
from lightning.random.random_fast cimport RandomState

cdef extern from "math.h":
   double fabs(double)

cdef extern from "float.h":
   double DBL_MAX


def _dual_cd(self,
             np.ndarray[double, ndim=1, mode='c'] w,
             np.ndarray[double, ndim=1, mode='c'] alpha,
             np.ndarray[double, ndim=2, mode='c'] X,
             np.ndarray[double, ndim=1]y,
             KernelCache kcache,
             int linear_kernel,
             selection,
             int search_size,
             termination,
             int n_components,
             double C,
             loss,
             int max_iter,
             RandomState rs,
             double tol,
             int shrinking,
             callback,
             int verbose):
    cdef Py_ssize_t n_samples = X.shape[0]
    cdef Py_ssize_t n_features = X.shape[1]

    cdef np.ndarray[int, ndim=1, mode='c'] A
    A = np.arange(n_samples, dtype=np.int32)
    cdef Py_ssize_t active_size = n_samples

    cdef double U
    cdef double D_ii

    if loss == "l1":
        U = C
        D_ii = 0
    elif loss == "l2":
        U = DBL_MAX
        D_ii = 1.0 / (2 * C)

    cdef np.ndarray[double, ndim=1, mode='c'] col
    col = np.zeros(n_samples, dtype=np.float64)

    cdef np.ndarray[double, ndim=1, mode='c'] Q_bar_diag
    Q_bar_diag = np.zeros(n_samples, dtype=np.float64)

    kcache.compute_diag(X, Q_bar_diag)
    Q_bar_diag += D_ii

    cdef double M
    cdef double m
    cdef int i, j
    cdef double y_i
    cdef double alpha_i, alpha_old
    cdef double M_bar = DBL_MAX
    cdef double m_bar = -DBL_MAX
    cdef unsigned int t = 0
    cdef int s
    cdef double G, PG
    cdef double step
    cdef int r

    cdef list[int].iterator it

    cdef int select_method = get_select_method(selection)
    cdef int check_n_sv = termination == "n_components"
    cdef int check_convergence = termination == "convergence"
    cdef int permute = selection == "permute" or linear_kernel
    cdef int has_callback = callback is not None
    cdef int stop = 0

    # FIXME: would be better to store the support indices in the class
    if not linear_kernel:
        for i in xrange(n_samples):
            if alpha[i] != 0:
                kcache.add_sv(i)

    for t in xrange(max_iter):
        if verbose >= 1:
            print "\nIteration", t

        if permute:
            rs.shuffle(A[:active_size])

        M = -DBL_MAX
        m = DBL_MAX

        s = 0
        while s < active_size:
            if permute:
                i = A[s]
            else:
                i = select_sv(A, search_size, active_size, select_method,
                              alpha, 0, X, y, kcache, col, 0, rs)

            y_i = y[i]
            alpha_i = fabs(alpha[i])

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
                kcache.compute_column_sv(X, X, i, col)
                it = kcache.support_set.begin()
                while it != kcache.support_set.end():
                    j = deref(it)
                    G += col[j] * y[i] * alpha[j]
                    inc(it)
                G += D_ii * alpha_i

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
                alpha_i = min(max(alpha_i - G / Q_bar_diag[i], 0), U)
                alpha[i] = alpha_i * y_i

                # Update support set.
                if not linear_kernel:
                    if alpha_i != 0:
                        kcache.add_sv(i)
                    elif alpha_i == 0:
                        kcache.remove_sv(i)

                if linear_kernel:
                    step = (alpha_i - alpha_old) * y_i
                    w += step * X[i]

            # Exit if necessary.
            if check_n_sv and kcache.n_sv() >= n_components:
                stop = 1
                break

            # Callback
            if has_callback and s % 100 == 0:
                ret = callback(self)
                if ret is not None:
                    stop = 1
                    break


            if verbose >= 1 and s % 100 == 0:
                sys.stdout.write(".")
                sys.stdout.flush()

            s += 1

        # end while

        if stop:
            break

        # Convergence check.
        if check_convergence and M - m <= tol:
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

    if linear_kernel:
        return w
    else:
        return alpha
