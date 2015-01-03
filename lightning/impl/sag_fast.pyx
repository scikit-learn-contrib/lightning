# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
#
# Author: Mathieu Blondel
# License: BSD

import numpy as np
cimport numpy as np

ctypedef np.int64_t LONG

from libc.math cimport sqrt

from lightning.impl.randomkit.random_fast cimport RandomState
from lightning.impl.dataset_fast cimport RowDataset
from lightning.impl.sgd_fast cimport LossFunction


cdef double _pred(double* data,
                  int* indices,
                  int n_nz,
                  double* w):

    cdef int j, jj
    cdef double dot = 0

    for jj in xrange(n_nz):
        j = indices[jj]
        dot += w[j] * data[jj]

    return dot


cdef void _add(double* data,
               int* indices,
               int n_nz,
               double scale,
               double* w):
    cdef int jj, j

    for jj in xrange(n_nz):
        j = indices[jj]
        w[j] += scale * data[jj]


def _sag_fit(self,
             RowDataset X,
             np.ndarray[double, ndim=1]y,
             np.ndarray[double, ndim=1]coef,
             np.ndarray[double, ndim=1]grad,
             double eta,
             double alpha,
             LossFunction loss,
             int max_iter,
             double tol,
             int verbose,
             callback,
             RandomState rng):

    cdef int n_samples = X.get_n_samples()
    cdef int n_features = X.get_n_features()

    # Variables.
    cdef int ii, i, jj, j, it, t
    cdef double y_pred, scale, g_old, tmp
    cdef double violation, violation_init, violation_ratio
    cdef double eta_avg = eta / n_samples
    cdef double eta_alpha = eta * alpha
    cdef int has_callback = callback is not None

    # Data pointers.
    cdef double* data
    cdef int* indices
    cdef int n_nz

    # Buffers and pointers.
    cdef np.ndarray[double, ndim=1] g_sum_
    g_sum_ = np.zeros(n_features, dtype=np.float64)
    cdef double* g_sum = <double*>g_sum_.data
    cdef double* w = <double*>coef.data
    cdef double* g = <double*>grad.data

    # Initialize gradient memory.
    for i in xrange(n_samples):
        # Retrieve sample i.
        X.get_row_ptr(i, &indices, &data, &n_nz)

        # Make prediction.
        y_pred = _pred(data, indices, n_nz, w)

        # A gradient is given by g[i] * X[i].
        g[i] = -loss.get_update(y_pred, y[i])

        # Update g_sum.
        _add(data, indices, n_nz, g[i], g_sum)

    # Outer loop.
    for it in xrange(max_iter):

        for ii in xrange(n_samples):
            i = rng.randint(n_samples - 1)

            # Retrieve sample i.
            X.get_row_ptr(i, &indices, &data, &n_nz)

            # Make prediction.
            y_pred = _pred(data, indices, n_nz, w)

            # Make copy of old gradient value.
            g_old = g[i]

            # A gradient is given by g[i] * X[i].
            g[i] = -loss.get_update(y_pred, y[i])

            # Update g_sum.
            _add(data, indices, n_nz, (g[i] - g_old), g_sum)

            # Update coefficients.
            for j in xrange(n_features):
                w[j] -= eta_alpha * w[j]
                w[j] -= eta_avg * g_sum[j]

        # Callback.
        if has_callback:
            ret = callback(self)
            if ret is not None:
                break

        # Compute optimality violation.
        violation = 0
        for j in xrange(n_features):
            tmp = g_sum[j] / n_samples + alpha * w[j]
            violation += tmp * tmp
        violation = sqrt(violation)

        # Convergence monitoring.
        if it == 0:
            violation_init = violation

        violation_ratio = violation / violation_init

        if verbose:
            print it + 1, violation_ratio

        if violation_ratio <= tol:
            if verbose:
                print "Converged"
            break
