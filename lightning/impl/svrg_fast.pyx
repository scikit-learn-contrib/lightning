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


def _svrg_fit(self,
              RowDataset X,
              np.ndarray[double, ndim=1]y,
              np.ndarray[double, ndim=1]coef,
              np.ndarray[double, ndim=1]coef_scale,
              np.ndarray[double, ndim=1]full_grad,
              np.ndarray[double, ndim=1]grad,
              double eta,
              double alpha,
              LossFunction loss,
              int max_iter,
              int n_inner,
              double tol,
              int verbose,
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
    cdef int has_callback = callback is not None

    # Data pointers.
    cdef double* data
    cdef int* indices
    cdef int n_nz

    # Buffers and pointers.
    cdef np.ndarray[int, ndim=1]last = np.zeros(n_features, dtype=np.int32)
    cdef double* w = <double*>coef.data
    cdef double* w_scale = <double*>coef_scale.data
    cdef double* fg = <double*>full_grad.data
    cdef double* g = <double*>grad.data

    for it in xrange(max_iter):

        # Reset full gradient
        for j in xrange(n_features):
            fg[j] = 0

        # Compute full gradient.
        for i in xrange(n_samples):

            # Retrieve sample i.
            X.get_row_ptr(i, &indices, &data, &n_nz)

            # Make prediction.
            y_pred = _pred(data, indices, n_nz, w) * w_scale[0]

            # A gradient is given by g[i] * X[i].
            g[i] = -loss.get_update(y_pred, y[i], i)

            _add(data, indices, n_nz, g[i], fg)

        # Compute optimality violation.
        violation = 0
        alpha_scaled = alpha * w_scale[0]
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

            # Add deterministic part, just in time.
            if t > 0:
                for jj in xrange(n_nz):
                    j = indices[jj]
                    w[j] -= eta_avg / w_scale[0] * (t - last[j]) * fg[j]
                    last[j] = t

            # Make prediction.
            y_pred = _pred(data, indices, n_nz, w) * w_scale[0]

            # A gradient is given by scale * X[i].
            scale = -loss.get_update(y_pred, y[i], i)

            w_scale[0] *= (1 - eta_alpha)

            # Add deterministic part.
            #for j in xrange(n_features):
                #w[j] -= eta_avg / w_scale * fg[j]

            # Add stochastic part.
            _add(data, indices, n_nz, eta * (g[i] - scale) / w_scale[0], w)

            # Take care of possible underflows.
            if w_scale[0] < 1e-9:
                for j in xrange(n_features):
                    w[j] *= w_scale[0]
                w_scale[0] = 1.0

        # Finalize.
        for j in xrange(n_features):
            w[j] -= eta_avg / w_scale[0] * (n_inner - last[j]) * fg[j]
            last[j] = 0

        # Callback.
        if has_callback:
            ret = callback(self)
            if ret is not None:
                break

    # Rescale coefficients.
    for j in xrange(n_features):
        w[j] *= w_scale[0]
    w_scale[0] = 1.0
