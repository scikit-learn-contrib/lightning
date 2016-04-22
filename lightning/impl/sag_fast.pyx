# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
#
# Authors: Mathieu Blondel
#          Fabian Pedregosa
#          Arnaud Rachez
# License: BSD

import numpy as np
cimport numpy as np

ctypedef np.int64_t LONG

from libc.math cimport sqrt, fabs

from lightning.impl.randomkit.random_fast cimport RandomState
from lightning.impl.dataset_fast cimport RowDataset
from lightning.impl.sgd_fast cimport LossFunction

# Reimplementation for MSVC support
cdef inline double fmax(double a, double b) nogil:
    return max(a, b)


cdef class Penalty:

    cdef void projection(self,
                         double* w,
                         int* indices,
                         double stepsize,
                         int n_nz):
        raise NotImplementedError()
    
    cdef void projection_lagged(self,
                                int t,
                                double* w,
                                double* g_sum,
                                int* indices,
                                double stepsize_prox,
                                double stepsize_grad,
                                double* lag_scaling,
                                int n_nz,
                                int* last,
                                double* scaling_seq):
        raise NotImplementedError()

    cdef double regularization(self, np.ndarray[double, ndim=1] coef):
        raise NotImplementedError()


cdef class L1Penalty(Penalty):

    def __cinit__(self):
        self.support_lagged = True

    cdef void projection(self,
                         double* w,
                         int* indices,
                         double stepsize,
                         int n_nz):

        cdef int j, jj

        for jj in xrange(n_nz):
            j = indices[jj]
            w[j] = fmax(w[j] - stepsize, 0) - fmax(-w[j] - stepsize, 0)

    cdef void projection_lagged(self,
                                int t,
                                double* w,
                                double* g_sum,
                                int* indices,
                                double stepsize_prox,
                                double stepsize_grad,
                                double* lag_scaling,
                                int n_nz,
                                int* last,
                                double* scaling_seq):
        cdef int i, j, jj
        cdef long missed_updates
        cdef double tmp

        for jj in range(n_nz):
            ind = indices[jj]
            missed_updates = t - last[ind]
            if fabs(g_sum[ind]) <= stepsize_prox:
                w[ind] -= stepsize_grad * lag_scaling[missed_updates] * g_sum[ind]
                w[ind] = fmax(w[ind] - lag_scaling[missed_updates] * stepsize_prox, 0) \
                        - fmax(-w[ind] - lag_scaling[missed_updates] * stepsize_prox, 0)
            else:
                for i in range(missed_updates, 0, -1):
                    w[ind] -= scaling_seq[i-1] * stepsize_grad * g_sum[ind]
                    tmp = stepsize_prox * scaling_seq[i-1]
                    w[ind] = fmax(w[ind] - tmp, 0) - fmax(-w[ind] - tmp, 0)
            last[ind] = t
        return

    cdef double regularization(self, np.ndarray[double, ndim=1] coef):

        cdef int j
        cdef double reg = 0

        for j in range(coef.size):
            reg += fabs(coef[j])
        return reg


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
                         double* lag_scaling,
                         int* indices,
                         int n_nz,
                         int* last,
                         double stepsize):
    """
    Apply missing updates to w, just-in-time. See [1, Section 4] 
    for a description of this technique.

    [1] 1. Schmidt, M., Roux, N. Le & Bach, F. Minimizing Finite 
        Sums with the Stochastic Average Gradient. 1–45 (2013).
    """
    cdef long missed_updates

    for jj in xrange(n_nz):
        ind = indices[jj]
        missed_updates = t - last[ind]
        w[ind] -= stepsize * lag_scaling[missed_updates] * g_sum[ind]
        last[ind] = t


def _sag_fit(self,
             RowDataset X,
             np.ndarray[double, ndim=1]y,
             np.ndarray[double, ndim=1]coef,
             np.ndarray[double, ndim=1]coef_scale,
             np.ndarray[double, ndim=1]grad,
             np.ndarray[double, ndim=1]sample_weight,
             double eta,
             double alpha,
             double beta,
             LossFunction loss,
             Penalty penalty,
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
    cdef double eta_alpha = eta * alpha
    if eta_alpha == 1.:
        # this is a problem because then w_scale[0]
        # becomes zero. Solution: decrease slightly eta
        eta = 0.9 * eta
        eta_alpha = eta * alpha

    cdef double eta_avg = eta / n_samples
    cdef double g_change = 0.
    cdef int has_callback = callback is not None

    # Data pointers.
    cdef double* data
    cdef int* indices
    cdef int n_nz

    # Buffers and pointers.
    cdef np.ndarray[int, ndim=1]last_ = np.zeros(n_features, dtype=np.int32)
    cdef np.ndarray[int, ndim=1] last_penalty_ = np.zeros(n_features, dtype=np.int32)
    cdef np.ndarray[double, ndim=1] g_sum_
    cdef np.ndarray[int, ndim=1] all_indices_ = np.arange(n_features, dtype=np.int32)
    g_sum_ = np.zeros(n_features, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] lag_scaling_
    lag_scaling_ = np.zeros(n_inner+2, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] scaling_seq_
    cdef double* scaling_seq

    cdef np.ndarray[double, ndim=1] w_violation_
    cdef double* w_violation
    cdef double* g_sum = <double*>g_sum_.data
    cdef double* w = <double*>coef.data
    cdef double* w_scale = <double*>coef_scale.data
    
    cdef double* g = <double*>grad.data
    cdef double* lag_scaling = <double*> lag_scaling_.data

    cdef int* last = <int*> last_.data
    cdef int* last_penalty_update = <int*> last_penalty_.data
    cdef int* all_indices = <int*> all_indices_.data
    cdef double geosum = 1.0
    cdef bint support_lagged = True
    cdef bint nontrivial_prox = saga and (penalty is not None)

    if nontrivial_prox:
        w_violation_ = np.zeros(n_features, dtype=np.float64)
        w_violation = <double*>w_violation_.data
        support_lagged = penalty.support_lagged
        if support_lagged:
            # L1 lagged updates requires to have the array of scalings
            scaling_seq_ = np.zeros(n_inner+1, dtype=np.float64)
            scaling_seq = <double*> scaling_seq_.data
            scaling_seq[0] = 1.

    lag_scaling[0] = 0.
    lag_scaling[1] = 1.
    for i in range(2, n_inner + 2):
        geosum *= (1 - eta_alpha)
        if nontrivial_prox and support_lagged:
            scaling_seq[i-1] = geosum
        lag_scaling[i] = lag_scaling[i-1] + geosum

    # Initialize gradient memory.
    for i in xrange(n_samples):
        # Retrieve sample i.
        X.get_row_ptr(i, &indices, &data, &n_nz)

        # Make prediction.
        y_pred = _pred(data, indices, n_nz, w) * w_scale[0]

        # A gradient is given by g[i] * X[i].
        g[i] = -sample_weight[i] * loss.get_update(y_pred, y[i])

        # Update g_sum.
        _add(data, indices, n_nz, g[i], g_sum)

    # Outer loop.
    for it in xrange(max_iter):

        # Inner loop.
        for t in xrange(n_inner):
            i = rng.randint(n_samples - 1)

            # Retrieve sample i.
            X.get_row_ptr(i, &indices, &data, &n_nz)

            # Apply missed updates.
            if t > 0 and support_lagged:
                if nontrivial_prox:
                    # SAGA with non-trivial prox
                    penalty.projection_lagged(
                        t, w, g_sum, indices, beta * eta / w_scale[0],
                        eta_avg / w_scale[0], lag_scaling, n_nz, last,
                        scaling_seq)
                else:
                    # SAG or SAGA with trivial prox
                    _lagged_update(t, w, g_sum, lag_scaling, indices,
                                   n_nz, last, eta_avg / w_scale[0])

            # Make prediction.
            y_pred = _pred(data, indices, n_nz, w) * w_scale[0]

            # Make copy of old gradient value.
            g_old = g[i]

            # A gradient is given by g[i] * X[i].
            g[i] = - sample_weight[i] * loss.get_update(y_pred, y[i])
            g_change = g[i] - g_old

            # Update coefficient scale (l2 regularization).
            w_scale[0] *= (1 - eta_alpha)
            # Take care of possible underflows.

            if w_scale[0] < 1e-9:
                for j in xrange(n_features):
                    w[j] *= w_scale[0]
                w_scale[0] = 1.0

            if saga:
                # update w with sparse step bit
                _add(data, indices, n_nz, -g_change * eta / w_scale[0], w)

                if support_lagged:
                    # gradient-average part of the step
                    _lagged_update(t + 1, w, g_sum, lag_scaling, indices,
                                   n_nz, last, eta_avg / w_scale[0])
                    if nontrivial_prox:
                        # prox update
                        penalty.projection(w, indices, beta * eta / w_scale[0],
                                           n_nz)
                else:
                    # gradient-average part of the step
                    # could be an _add instead of a _lagged update since we are not
                    # using the last array anywhere else
                    _lagged_update(t + 1, w, g_sum, lag_scaling, all_indices,
                                   n_features, last, eta_avg / w_scale[0])
                    if nontrivial_prox:
                        # prox update
                        penalty.projection(w, all_indices, beta * eta / w_scale[0],
                                           n_features)

            # Update g_sum.
            _add(data, indices, n_nz, g_change, g_sum)

        # Finalize.
        if support_lagged:
            if nontrivial_prox:
                penalty.projection_lagged(
                    n_inner, w, g_sum, all_indices, beta * eta / w_scale[0],
                    eta_avg / w_scale[0], lag_scaling, n_features, last,
                    scaling_seq)
            else:
                _lagged_update(n_inner, w, g_sum, lag_scaling, all_indices,
                               n_features, last, eta_avg / w_scale[0])

        for j in range(n_features):
            last[j] = 0
            last_penalty_update[j] = 0

        # Callback.
        if has_callback:
            ret = callback(self)
            if ret is not None:
                break

        # Compute optimality violation.
        violation = 0
        alpha_scaled = alpha * w_scale[0]
        if nontrivial_prox:
            for j in xrange(n_features):
                    w_violation[j] = w_scale[0] * w[j] - \
                            eta * (g_sum[j] / n_samples + alpha_scaled * w[j])

            penalty.projection(w_violation, all_indices, beta * eta / w_scale[0],
                               n_features)

            for j in xrange(n_features):
                violation += (w_scale[0] * w[j] - w_violation[j])**2

        else:
            for j in xrange(n_features):
                tmp = g_sum[j] / n_samples + alpha_scaled * w[j]
                violation += tmp * tmp

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
            print(it + 1, violation_ratio)

        if violation_ratio <= tol:
            if verbose:
                print("Converged")
            break

    # Rescale coefficients.
    for j in xrange(n_features):
        w[j] *= w_scale[0]
    w_scale[0] = 1.0
