# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
#
# Author: Mathieu Blondel
# License: BSD


cimport numpy as np


cdef extern from "math.h":
   double exp(double)


cdef double powi(double base, int times):
    cdef double tmp = base, ret = 1.0

    cdef int t = times

    while t > 0:
        if t % 2 == 1:
            ret *= tmp
        tmp = tmp * tmp

        t /= 2

    return ret


cdef class Kernel:

    cpdef double compute(self,
                         np.ndarray[double, ndim=2, mode='c'] X,
                         int i,
                         np.ndarray[double, ndim=2, mode='c'] Y,
                         int j):
        raise NotImplementedError()

    cpdef double compute_self(self,
                              np.ndarray[double, ndim=2, mode='c'] X,
                              int i):
        return self.compute(X, i, X, i)

    cpdef compute_diag(self,
                       np.ndarray[double, ndim=2, mode='c'] X,
                       np.ndarray[double, ndim=1, mode='c'] out):
        self.compute_diag_ptr(X, <double*>out.data)

    cdef void compute_diag_ptr(self,
                               np.ndarray[double, ndim=2, mode='c'] X,
                               double* out):

        cdef Py_ssize_t n_samples = X.shape[0]
        cdef int i

        for i in xrange(n_samples):
            out[i] = self.compute_self(X, i)

    cpdef compute_column(self,
                         np.ndarray[double, ndim=2, mode='c'] X,
                         np.ndarray[double, ndim=2, mode='c'] Y,
                         int j,
                         np.ndarray[double, ndim=1, mode='c'] out):
        self.compute_column_ptr(X, Y, j, <double*>out.data)

    cdef void compute_column_ptr(self,
                                 np.ndarray[double, ndim=2, mode='c'] X,
                                 np.ndarray[double, ndim=2, mode='c'] Y,
                                 int j,
                                 double* out):

        cdef Py_ssize_t n_samples = X.shape[0]
        cdef int i

        for i in xrange(n_samples):
            out[i] = self.compute(X, i, Y, j)


cdef class LinearKernel(Kernel):

    cpdef double compute(self,
                         np.ndarray[double, ndim=2, mode='c'] X,
                         int i,
                         np.ndarray[double, ndim=2, mode='c'] Y,
                         int j):

        cdef Py_ssize_t n_features = X.shape[1]

        cdef double dot = 0
        cdef int k

        for k in xrange(n_features):
            dot += X[i, k] * Y[j, k]

        return dot


cdef class PolynomialKernel(Kernel):

    def __init__(self, int degree, double coef0, double gamma):
        self.degree = degree
        self.coef0 = coef0
        self.gamma = gamma

    cpdef double compute(self,
                         np.ndarray[double, ndim=2, mode='c'] X,
                         int i,
                         np.ndarray[double, ndim=2, mode='c'] Y,
                         int j):

        cdef Py_ssize_t n_features = X.shape[1]

        cdef double dot = 0
        cdef int k

        for k in xrange(n_features):
            dot += X[i, k] * Y[j, k]

        return powi(self.coef0 + dot * self.gamma, self.degree)


cdef class RbfKernel(Kernel):

    def __init__(self, double gamma):
        self.gamma = gamma

    cpdef double compute(self,
                         np.ndarray[double, ndim=2, mode='c'] X,
                         int i,
                         np.ndarray[double, ndim=2, mode='c'] Y,
                         int j):

        cdef Py_ssize_t n_features = X.shape[1]

        cdef double diff, value = 0
        cdef int k

        for k in xrange(n_features):
            diff = X[i, k] - Y[j, k]
            value += diff * diff

        return exp(-self.gamma * value)

    cpdef double compute_self(self,
                              np.ndarray[double, ndim=2, mode='c'] X,
                              int i):
        return 1.0


cdef class PrecomputedKernel(Kernel):

    cpdef double compute(self,
                         np.ndarray[double, ndim=2, mode='c'] X,
                         int i,
                         np.ndarray[double, ndim=2, mode='c'] Y,
                         int j):
        return X[i, j]
