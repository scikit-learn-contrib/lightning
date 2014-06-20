# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
#
# Author: Mathieu Blondel
# License: BSD

import numpy as np
cimport numpy as np

from lightning.impl.random.random_fast cimport RandomState
from lightning.impl.dataset_fast cimport RowDataset


cdef int _predict(double dot,
                  np.ndarray[double, ndim=1, mode='c'] b,
                  int n_classes):
    cdef int r
    cdef int y_hat = 0

    for r in xrange(n_classes):
        if dot - b[r] < 0:
            y_hat = r
            break

    return y_hat


def _prank_fit(np.ndarray[double, ndim=1, mode='c'] w,
               np.ndarray[double, ndim=1, mode='c'] b,
               RowDataset X,
               np.ndarray[int, ndim=1] y,
               int n_classes,
               int n_iter,
               RandomState rs,
               int shuffle):

    cdef int n_samples = X.get_n_samples()
    cdef int n_features = X.get_n_features()

    cdef int n, i, ii, j, jj, y_hat, tau, yr, r
    cdef double dot

    # Data pointers.
    cdef double* data
    cdef int* indices
    cdef int n_nz

    # Data indices.
    cdef np.ndarray[int, ndim=1] ind
    ind = np.arange(n_samples, dtype=np.int32)

    for n in xrange(n_iter):
        if shuffle:
            rs.shuffle(ind)

        for ii in xrange(n_samples):
            i = ind[ii]

            # Retrieve row.
            X.get_row_ptr(i, &indices, &data, &n_nz)

            # Compute dot product.
            dot = 0
            for jj in xrange(n_nz):
                j = indices[jj]
                dot += w[j] * data[jj]

            y_hat = _predict(dot, b, n_classes)

            # Nothing to do if prediction was correct.
            if y_hat == y[i]:
                continue

            tau = 0
            for r in xrange(n_classes - 1):
                if y[i] <= r:
                    yr = -1
                else:
                    yr = 1

                if yr * (dot - b[r]) <= 0:
                    tau += yr
                    b[r] -= yr

            # Update w.
            for jj in xrange(n_nz):
                j = indices[jj]
                w[j] += tau * data[jj]


def _prank_predict(np.ndarray[double, ndim=1, mode='c'] dot,
                   np.ndarray[double, ndim=1, mode='c'] b,
                   int n_classes,
                   np.ndarray[int, ndim=1, mode='c'] out):

    cdef int n_samples = dot.shape[0]
    cdef int i

    for i in xrange(n_samples):
        out[i] = _predict(dot[i], b, n_classes)
