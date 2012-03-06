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
   double pow(double,double)


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


cdef class PrecomputedKernel(Kernel):

    def __init__(self, np.ndarray[double, ndim=2, mode='c'] K):
        self.K = <double*>K.data

    cpdef double compute(self,
                         np.ndarray[double, ndim=2, mode='c'] X,
                         int i,
                         np.ndarray[double, ndim=2, mode='c'] Y,
                         int j):
        cdef Py_ssize_t n_samples = X.shape[0]
        cdef int k = i * n_samples + j
        return self.K[k]
