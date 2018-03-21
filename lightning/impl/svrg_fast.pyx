# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
#
# Author: Mathieu Blondel
#         Krishna Pillutla (averaging support)
# License: BSD

import numpy as np
cimport numpy as np

ctypedef np.int64_t LONG

from libc.math cimport sqrt
from libc.math cimport pow as powc

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


def _svrg_fit(self,
              RowDataset X,
              np.ndarray[double, ndim=1]y,
              np.ndarray[double, ndim=1]coef,
              np.ndarray[double, ndim=1]avg_coef,
              np.ndarray[double, ndim=1]full_grad,
              np.ndarray[double, ndim=1]grad,
              double eta,
              double alpha,
              LossFunction loss,
              int max_iter,
              int n_inner,
              double tol,
              int verbose,
              int do_averaging,
              callback,
              RandomState rng):

    cdef int n_samples = X.get_n_samples()
    cdef int n_features = X.get_n_features()

    # Variables.
    cdef int i, jj, j, it, t
    cdef double y_pred, scale, tmp, alpha_scaled
    cdef double violation, violation_init, violation_ratio
    cdef double eta_avg = eta / n_samples
    cdef double eta_alpha = eta * alpha
    cdef double one_minus_eta_alpha = 1 - eta_alpha
    cdef double one_over_eta_alpha = 1 / eta_alpha if eta_alpha > 0 else 0.0
    cdef int has_callback = callback is not None
    cdef double w_scale = 1.0
    cdef double avg_a = 0.0, avg_b = 1.0
    cdef double correction, correction_avg
    cdef double mu

    # Data pointers.
    cdef double* data
    cdef int* indices
    cdef int n_nz

    # Buffers and pointers.
    cdef double* w = <double*>coef.data
    cdef double* w_avg = <double*>avg_coef.data
    cdef double* fg = <double*>full_grad.data
    cdef double* g = <double*>grad.data

    for it in xrange(max_iter):

        # Reset full gradient.
        for j in xrange(n_features):
            fg[j] = 0

        # Compute full gradient.
        for i in xrange(n_samples):

            # Retrieve sample i.
            X.get_row_ptr(i, &indices, &data, &n_nz)

            # Make prediction.
            y_pred = _pred(data, indices, n_nz, w) * w_scale

            # A gradient is given by g[i] * X[i].
            g[i] = -loss.get_update(y_pred, y[i])

            _add(data, indices, n_nz, g[i], fg)

        # Compute optimality violation.
        violation = 0
        alpha_scaled = alpha * w_scale
        for j in xrange(n_features):
            tmp = fg[j] / n_samples + alpha_scaled * w[j]
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

        # Inner loop.
        for t in xrange(n_inner):
            i = rng.randint(n_samples - 1)

            # Retrieve sample i.
            X.get_row_ptr(i, &indices, &data, &n_nz)

            # Make prediction, accounting for correction due to
            # dense (deterministic) part of update.
            y_pred = _pred(data, indices, n_nz, w) * w_scale
            if eta_alpha > 0:
                correction = (1 - powc(one_minus_eta_alpha, t)) / eta_alpha
            else:
                correction = t
            y_pred -= _pred(data, indices, n_nz, fg) * eta_avg * correction

            # A gradient is given by scale * X[i].
            scale = -loss.get_update(y_pred, y[i])

            w_scale *= (1 - eta_alpha)

            # Add stochastic part.
            _add(data, indices, n_nz, eta * (g[i] - scale) / w_scale, w)

            # Update average (or reset, at t = 0) of stochastic part.
            if t == 0:
                for j in xrange(n_features):
                    w_avg[j] = 0.0
                avg_a = w_scale
                avg_b = 1.0
            else:
                mu = 1.0 / (t + 1)
                _add(data, indices, n_nz, eta * (scale - g[i]) * avg_a / w_scale, w_avg)
                avg_b /= (1.0 - mu)
                avg_a += mu * avg_b * w_scale

            # Take care of possible underflows.
            if w_scale < 1e-9:
                for j in xrange(n_features):
                    w[j] *= w_scale
                avg_a /= w_scale
                w_scale = 1.0

        # Finalize. Reconstruct w and w_avg. Add deterministic update to w and w_avg.
        if eta_alpha > 0:
            correction = (1.0 - powc(one_minus_eta_alpha, n_inner)) / eta_alpha
            correction_avg = one_over_eta_alpha - one_minus_eta_alpha * correction / (n_inner * eta_alpha)
        else:
            correction = n_inner
            correction_avg = (n_inner - 1.0) / 2.0
        for j in xrange(n_features):
            w_avg[j] = (w_avg[j] + avg_a * w[j]) / avg_b
            w_avg[j] -= eta_avg * fg[j] * correction_avg
            w[j] *= w_scale
            w[j] -= eta_avg * fg[j] * correction
        w_scale = 1.0
        avg_a = 0.0
        avg_b = 1.0

        # Update iterate, if averaging
        if do_averaging:
            for j in xrange(n_features):
                w[j] = w_avg[j]

        # Callback.
        if has_callback:
            ret = callback(self)
            if ret is not None:
                break
