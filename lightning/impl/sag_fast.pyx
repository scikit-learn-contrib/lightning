# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
#
# Authors: Mathieu Blondel
#          Fabian Pedregosa
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

cdef void _lagged_update(int t,
                         double* w,
                         double* g_sum,
                         double* scale_cumm,
                         int* indices,
                         double w_scale,
                         int n_nz,
                         int* last,
                         double eta_avg):
    """
    Apply missing updates to w, just-in-time.
    """
    cdef double incr_scale
    scale_cumm[t] = scale_cumm[t-1] + (1./w_scale)
    for jj in xrange(n_nz):
        j = indices[jj]
        incr_scale = scale_cumm[t] - scale_cumm[last[j]]
        w[j] -= eta_avg * incr_scale * g_sum[j]
        last[j] = t


def _sag_fit(self,
             RowDataset X,
             np.ndarray[double, ndim=1]y,
             np.ndarray[double, ndim=1]coef,
             np.ndarray[double, ndim=1]coef_scale,
             np.ndarray[double, ndim=1]grad,
             double eta,
             double alpha,
             LossFunction loss,
             int max_iter,
             int n_inner,
             double tol,
             int verbose,
             callback,
             RandomState rng,
             bint saga):

    cdef int n_samples = X.get_n_samples()
    cdef int n_features = X.get_n_features()

    # Variables.
    cdef int i, jj, j, it, t
    cdef double y_pred, scale, g_old, tmp, alpha_scaled
    cdef double violation, violation_init, violation_ratio
    cdef double eta_avg = eta / n_samples
    cdef double eta_alpha = eta * alpha
    cdef double g_change = 0.
    cdef int has_callback = callback is not None

    # Data pointers.
    cdef double* data
    cdef int* indices
    cdef int n_nz

    # Buffers and pointers.
    cdef np.ndarray[int, ndim=1]last_ = np.zeros(n_features, dtype=np.int32)
    cdef np.ndarray[double, ndim=1] g_sum_
    cdef np.ndarray[int, ndim=1] all_indices_ = np.arange(n_features, dtype=np.int32)
    g_sum_ = np.zeros(n_features, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] scale_cumm_
    scale_cumm_ = np.zeros(n_inner+2, dtype=np.float64)
    cdef double* g_sum = <double*>g_sum_.data
    cdef double* w = <double*>coef.data
    cdef double* w_scale = <double*>coef_scale.data
    cdef double* g = <double*>grad.data
    cdef double* scale_cumm = <double*> scale_cumm_.data
    cdef int* last = <int*> last_.data
    cdef int* all_indices = <int*> all_indices_.data

    # Initialize gradient memory.
    for i in xrange(n_samples):
        # Retrieve sample i.
        X.get_row_ptr(i, &indices, &data, &n_nz)

        # Make prediction.
        y_pred = _pred(data, indices, n_nz, w) * w_scale[0]

        # A gradient is given by g[i] * X[i].
        g[i] = -loss.get_update(y_pred, y[i])

        # Update g_sum.
        _add(data, indices, n_nz, g[i], g_sum)

    # Outer loop.
    for it in xrange(max_iter):

        # Inner loop.
        for t in xrange(n_inner):
            i = rng.randint(n_samples - 1)

            # Retrieve sample i.
            X.get_row_ptr(i, &indices, &data, &n_nz)

            # Apply missed updates, just in time.
            if t > 0:
                _lagged_update(t, w, g_sum, scale_cumm, indices,
                               w_scale[0], n_nz, last, eta_avg)

            # Make prediction.
            y_pred = _pred(data, indices, n_nz, w) * w_scale[0]

            # Make copy of old gradient value.
            g_old = g[i]

            # A gradient is given by g[i] * X[i].
            g[i] = -loss.get_update(y_pred, y[i])
            g_change = g[i] - g_old

            # Update coefficient scale (l2 regularization).
            w_scale[0] *= (1 - eta_alpha)
            if saga:
                # update w with sparse step bit
                _add(data, indices, n_nz, g_change * eta_avg, w)

                ## gradient-average part of the step
                _lagged_update(t+1, w, g_sum, scale_cumm, indices,
                               w_scale[0], n_nz, last, eta_avg)

                # prox step
                # XXX TODO

            # Update g_sum.
            _add(data, indices, n_nz, g_change, g_sum)

            # Take care of possible underflows.
            if w_scale[0] < 1e-9:
                for j in xrange(n_features):
                    if last[j] != t:
                        # need to update the coefficient
                        tmp = scale_cumm[t] - scale_cumm[last[j]]
                        w[j] -= eta_avg * tmp * g_sum[j]
                        last[j] = t
                    w[j] *= w_scale[0]
                w_scale[0] = 1.0

        # Finalize.
        _lagged_update(n_inner, w, g_sum, scale_cumm, all_indices,
                       w_scale[0], n_features, last, eta_avg)
        for j in range(n_features):
            last[j] = 0

        # Callback.
        if has_callback:
            ret = callback(self)
            if ret is not None:
                break

        # Compute optimality violation.
        violation = 0
        alpha_scaled = alpha * w_scale[0]
        for j in xrange(n_features):
            tmp = g_sum[j] / n_samples + alpha_scaled * w[j]
            violation += tmp * tmp
        violation = sqrt(violation)

        # Convergence monitoring.
        if it == 0:
            if violation != 0:
                violation_init = violation
            else:
                # First epoch is optimal.  Setting violation_init to a positive
                # value to avoid division by zero.
                violation_init = 1.0

        violation_ratio = violation / violation_init

        if verbose:
            print it + 1, violation_ratio

        if violation_ratio <= tol:
            if verbose:
                print "Converged"
            break

    # Rescale coefficients.
    for j in xrange(n_features):
        w[j] *= w_scale[0]
    w_scale[0] = 1.0
