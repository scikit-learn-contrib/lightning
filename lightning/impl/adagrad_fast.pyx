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


cdef double _proj_elastic(double eta,
                          LONG t,
                          double g_sum,
                          double g_norm,
                          double alpha1,
                          double alpha2,
                          double delta):

    cdef double eta_t = eta * t
    cdef double denom = (delta + sqrt(g_norm) + eta_t * alpha2)
    cdef double wj_new1 = eta_t * (-g_sum / t - alpha1) / denom
    cdef double wj_new2 = eta_t * (-g_sum / t + alpha1) / denom

    if wj_new1 > 0:
        return wj_new1
    elif wj_new2 < 0:
        return wj_new2
    else:
        return 0


cpdef double _proj_elastic_all(double eta,
                               LONG t,
                               np.ndarray[double, ndim=1] g_sum,
                               np.ndarray[double, ndim=1] g_norms,
                               double alpha1,
                               double alpha2,
                               double delta,
                               np.ndarray[double, ndim=1] w):
    cdef int n_features = w.shape[0]
    cdef int j
    for j in xrange(n_features):
        w[j] = _proj_elastic(eta, t, g_sum[j], g_norms[j], alpha1, alpha2,
                             delta)


def _adagrad_fit(self,
                 RowDataset X,
                 np.ndarray[double, ndim=1]y,
                 np.ndarray[double, ndim=1]coef,
                 np.ndarray[double, ndim=1]g_sum,
                 np.ndarray[double, ndim=1]g_norms,
                 LossFunction loss,
                 double eta,
                 double delta,
                 double alpha1,
                 double alpha2,
                 int n_iter,
                 callback,
                 int n_calls,
                 rng):

    cdef int n_samples = X.get_n_samples()
    cdef int n_features = X.get_n_features()

    # Variables
    cdef LONG t
    cdef int it, ii, i, jj, j
    cdef double y_pred, tmp, scale
    cdef np.ndarray[int, ndim=1] sindices
    sindices = np.arange(n_samples, dtype=np.int32)
    cdef int has_callback = callback is not None

    # Data pointers.
    cdef double* data
    cdef int* indices
    cdef int n_nz

    # Pointers
    cdef double* w = <double*>coef.data

    t = 1
    for t in xrange(n_iter):

        # Shuffle sample indices.
        rng.shuffle(sindices)

        for ii in xrange(n_samples):
            i = sindices[ii]

            # Retrieve sample i.
            X.get_row_ptr(i, &indices, &data, &n_nz)

            # Update w lazily.
            if t > 1:
                for jj in xrange(n_nz):
                    j = indices[jj]
                    w[j] = _proj_elastic(eta, t - 1, g_sum[j], g_norms[j],
                                         alpha1, alpha2, delta)

            # Make prediction.
            y_pred = _pred(data, indices, n_nz, w)

            # A subgradient is given by scale * X[i].
            scale = -loss.get_update(y_pred, y[i])

            # Update g_sum and g_norms.
            if scale != 0:
                for jj in xrange(n_nz):
                    j = indices[jj]
                    tmp = scale * data[jj]
                    g_sum[j] += tmp
                    g_norms[j] += tmp * tmp

            # Update w by naive implementation: very slow.
            # for j in xrange(n_features):
            #    w[j] = _proj_elastic(eta, t, g_sum[j], g_norms[j], alpha1,
            #                         alpha2, delta)

            # Callback.
            if has_callback and t % n_calls == 0:
                ret = callback(self, t)
                if ret is not None:
                    break

            t += 1


    # Finalize.
    _proj_elastic_all(eta, t - 1, g_sum, g_norms, alpha1, alpha2, delta, coef)
