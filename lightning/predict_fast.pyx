# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
#
# Author: Mathieu Blondel
# License: BSD

cdef extern from "float.h":
   double DBL_MAX

import numpy as np
cimport numpy as np

cpdef decision_function_alpha(np.ndarray[double, ndim=2, mode='c'] X,
                              np.ndarray[double, ndim=2, mode='c'] sv,
                              np.ndarray[double, ndim=2, mode='c'] alpha,
                              Kernel kernel,
                              np.ndarray[double, ndim=2, mode='c'] out):

    cdef Py_ssize_t n_samples = X.shape[0]
    cdef Py_ssize_t n_sv = sv.shape[0]
    cdef Py_ssize_t n_vectors = alpha.shape[0]
    cdef double kvalue

    cdef int i, j, k

    if n_vectors == 1:
        for i in xrange(n_sv):
            if alpha[0, i] != 0:
                for j in xrange(n_samples):
                    out[j, 0] += alpha[0, i] * kernel.compute(X, j, sv, i)
    else:
        for i in xrange(n_sv):
            for j in xrange(n_samples):
                kvalue = kernel.compute(X, j, sv, i)
                for k in xrange(n_vectors):
                    out[j, k] += alpha[k, i] * kvalue


cpdef predict_alpha(np.ndarray[double, ndim=2, mode='c'] X,
                    np.ndarray[double, ndim=2, mode='c'] sv,
                    np.ndarray[double, ndim=2, mode='c'] alpha,
                    np.ndarray[int, ndim=1, mode='c'] classes,
                    Kernel kernel,
                    np.ndarray[double, ndim=1, mode='c'] out):

    cdef Py_ssize_t n_samples = X.shape[0]
    cdef Py_ssize_t n_sv = sv.shape[0]
    cdef Py_ssize_t n_classes = classes.shape[0]
    cdef np.ndarray[double, ndim=2, mode='c'] out2
    cdef double max_
    cdef int selected
    cdef int i, j

    if n_classes == 2:
        for i in xrange(n_sv):
            if alpha[0, i] != 0:
                for j in xrange(n_samples):
                    out[j] += alpha[0, i] * kernel.compute(X, j, sv, i)

        for j in xrange(n_samples):
            if out[j] > 0:
                out[j] = classes[1]
            elif out[j] < 0:
                out[j] = classes[0]
    else:
        out2 = np.zeros((n_samples, n_classes), dtype=np.float64)
        decision_function_alpha(X, sv, alpha, kernel, out2)

        for i in xrange(n_samples):

            selected = 0
            max_ = -DBL_MAX

            for j in xrange(n_classes):
                if out2[i, j] > max_:
                    max_ = out2[i, j]
                    selected = j

            out[i] = classes[selected]
