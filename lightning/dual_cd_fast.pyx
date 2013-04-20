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

from lightning.select_fast cimport get_select_method, select_sv
from lightning.random.random_fast cimport RandomState
from lightning.dataset_fast cimport Dataset
from lightning.dataset_fast cimport KernelDataset

cdef extern from "math.h":
   double fabs(double)

cdef extern from "float.h":
   double DBL_MAX


def _dual_cd(self,
             np.ndarray[double, ndim=1, mode='c'] w,
             np.ndarray[double, ndim=1, mode='c'] alpha,
             Dataset X,
             KernelDataset kds,
             np.ndarray[double, ndim=1]y,
             int linear_kernel,
             selection,
             int search_size,
             int permute,
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
    cdef list[int].iterator it
    cdef int select_method = get_select_method(selection)
    cdef int check_n_sv = termination == "n_components"
    cdef int check_convergence = termination == "convergence"
    cdef int cyclic = selection == "cyclic" or linear_kernel
    cdef int has_callback = callback is not None
    cdef int stop = 0
    cdef double U
    cdef double D_ii

    # Loss-dependent values.
    if loss == "l1":
        U = C
        D_ii = 0
    elif loss == "l2":
        U = DBL_MAX
        D_ii = 1.0 / (2 * C)

    # Active set.
    cdef np.ndarray[int, ndim=1, mode='c'] A
    A = np.arange(n_samples, dtype=np.int32)
    cdef Py_ssize_t active_size = n_samples

    # Diagonal values of the Q matrix.
    cdef np.ndarray[double, ndim=1, mode='c'] Q_bar_diag
    Q_bar_diag = np.zeros(n_samples, dtype=np.float64)
    kds.get_diag_out(<double*>Q_bar_diag.data)
    Q_bar_diag += D_ii

    # Data pointers.
    cdef double* data
    cdef int* indices
    cdef int n_nz

    # FIXME: would be better to store the support indices in the class
    if not linear_kernel:
        for i in xrange(n_samples):
            if alpha[i] != 0:
                kds.add_sv(i)

    for t in xrange(max_iter):
        if verbose >= 1:
            print "\nIteration", t

        if permute:
            rs.shuffle(A[:active_size])

        M = -DBL_MAX
        m = DBL_MAX

        s = 0
        while s < active_size:
            if cyclic:
                i = A[s]
            else:
                i = select_sv(A, search_size, active_size, select_method,
                              alpha, 0, kds, y, 0, rs)

            y_i = y[i]
            alpha_i = fabs(alpha[i])

            # Compute ith element of the gradient.
            if linear_kernel:
                X.get_row_ptr(i, &indices, &data, &n_nz)

                # G = y_i * np.dot(w, X[i]) - 1 + D_ii * alpha_i
                G = 0
                for jj in xrange(n_nz):
                    j = indices[jj]
                    G += w[j] * data[jj]
                G = y_i * G - 1 + D_ii * alpha_i
            else:
                # G = np.dot(Q_bar, alpha)[i] - 1
                G = -1
                col = kds.get_column_sv_ptr(i)
                it = kds.support_set.begin()
                while it != kds.support_set.end():
                    j = deref(it)
                    G += col[j] * y[i] * alpha[j]
                    inc(it)
                G += D_ii * alpha_i

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

                # Update support set.
                if not linear_kernel:
                    if alpha_i != 0:
                        kds.add_sv(i)
                    elif alpha_i == 0:
                        kds.remove_sv(i)

                if linear_kernel:
                    # Update the primal coefficients.
                    step = (alpha_i - alpha_old) * y_i
                    for jj in xrange(n_nz):
                        j = indices[jj]
                        w[j] += step * data[jj]

            # Exit if necessary.
            if check_n_sv and kds.n_sv() >= n_components:
                stop = 1
                break

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

    return w, alpha
