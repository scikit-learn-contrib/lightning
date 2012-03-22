# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
#
# Author: Mathieu Blondel
# License: BSD

from cython.operator cimport dereference as deref
from cython.operator cimport preincrement as inc
from cython.operator cimport postincrement as postinc
from cython.operator cimport predecrement as dec

from libcpp.list cimport list
from libcpp.map cimport map
from libc cimport stdlib

import numpy as np
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


cdef class KernelCache(Kernel):

    def __init__(self, Kernel kernel, int n_samples, int capacity, int verbose):
        self.kernel = kernel
        self.n_samples = n_samples
        self.capacity = capacity
        self.verbose = verbose
        self.size = 0

    def __cinit__(self, Kernel kernel, int n_samples, int capacity, int verbose):
        cdef int i

        self.support_set = new list[int]()
        self.support_vector = <int*> stdlib.malloc(sizeof(int) * n_samples)
        self.support_it = <list[int].iterator*> \
            stdlib.malloc(sizeof(list[int].iterator) * n_samples)

        self.n_computed = <int*> stdlib.malloc(sizeof(int) * n_samples)
        self.columns = new map[int, double*]()

        for i in xrange(n_samples):
            self.support_vector[i] = -1
            self.n_computed[i] = 0

    def __dealloc__(self):
        del self.support_set
        stdlib.free(self.support_vector)
        stdlib.free(self.support_it)

        self._clear_columns(self.n_samples)
        self.columns.clear()
        del self.columns
        stdlib.free(self.n_computed)

    cpdef double compute(self,
                         np.ndarray[double, ndim=2, mode='c'] X,
                         int i,
                         np.ndarray[double, ndim=2, mode='c'] Y,
                         int j):
        return self.kernel.compute(X, i, Y, j)

    cpdef double compute_self(self,
                              np.ndarray[double, ndim=2, mode='c'] X,
                              int i):
        return self.kernel.compute_self(X, i)

    cdef _create_column(self, int i):
        cdef int n_computed = self.n_computed[i]

        if n_computed != 0:
            return

        cdef int col_size = self.n_samples * sizeof(double)

        if self.size + col_size > self.capacity:
            if self.verbose >= 1:
                print "Half cache"
            self._clear_columns(self.columns.size() / 2)

        self.columns[0][i] = <double*> stdlib.calloc(self.n_samples,
                                                     sizeof(double))
        self.size += col_size

    cdef _clear_columns(self, int n):
        cdef map[int, double*].iterator it
        it = self.columns.begin()
        cdef int i = 0
        cdef int col_size

        while it != self.columns.end():
            col_size = self.n_samples * sizeof(double)
            stdlib.free(deref(it).second)
            self.n_computed[deref(it).first] = 0
            self.size -= col_size
            self.columns.erase(postinc(it))

            if i >= n - 1:
                break

            i += 1

    cpdef compute_diag(self,
                       np.ndarray[double, ndim=2, mode='c'] X,
                       np.ndarray[double, ndim=1, mode='c'] out):
        cdef Py_ssize_t n_samples = X.shape[0]
        cdef int i

        for i in xrange(n_samples):
            out[i] = self.kernel.compute_self(X, i)


    cpdef compute_column(self,
                         np.ndarray[double, ndim=2, mode='c'] X,
                         np.ndarray[double, ndim=2, mode='c'] Y,
                         int j,
                         np.ndarray[double, ndim=1, mode='c'] out):

        cdef int i = 0

        if self.capacity == 0:
            for i in xrange(self.n_samples):
                out[i] = self.kernel.compute(X, i, Y, j)
            return

        cdef int n_computed = self.n_computed[j]

        self._create_column(j)

        cdef double* cache = &(self.columns[0][j][0])

        if n_computed == -1:
            # Full column is already computed.
            for i in xrange(self.n_samples):
                out[i] = cache[i]
        elif n_computed > 0:
            # Some elements are already computed.
            for i in xrange(self.n_samples):
                if cache[i] == 0:
                    out[i] = self.kernel.compute(X, i, Y, j)
                    cache[i] = out[i]
                else:
                    out[i] = cache[i]
        else:
            # All elements must be computed.
            for i in xrange(self.n_samples):
                out[i] = self.kernel.compute(X, i, Y, j)
                cache[i] = out[i]

        self.n_computed[j] = -1


    cpdef compute_column_sv(self,
                            np.ndarray[double, ndim=2, mode='c'] X,
                            np.ndarray[double, ndim=2, mode='c'] Y,
                            int j,
                            np.ndarray[double, ndim=1, mode='c'] out):

        cdef int s
        cdef list[int].iterator it
        cdef int n_computed = self.n_computed[j]
        cdef int ssize = self.support_set.size()

        if ssize == 0:
            return

        if self.capacity == 0:
            it = self.support_set.begin()
            while it != self.support_set.end():
                s = deref(it)
                out[s] = self.kernel(X, s, Y, j)
                inc(it)
            return

        self._create_column(j)
        cdef double* cache = &(self.columns[0][j][0])

        if n_computed == -1:
            it = self.support_set.begin()
            while it != self.support_set.end():
                s = deref(it)
                out[s] = cache[s]
                inc(it)
            return

        it = self.support_set.begin()
        while it != self.support_set.end():
            s = deref(it)
            if cache[s] == 0:
                out[s] = self.kernel.compute(X, s, Y, j)
                cache[s] = out[s]
            else:
                out[s] = cache[s]

            inc(it)

        self.n_computed[j] = ssize

    cpdef remove_column(self, int i):
        if self.verbose >= 2:
            print "Remove column SV", i

        cdef map[int, double*].iterator it
        cdef int col_size = self.n_samples * sizeof(double)

        it = self.columns.find(i)

        if it != self.columns.end():
            self.n_computed[deref(it).first] = 0
            stdlib.free(deref(it).second)
            self.columns.erase(it)
            self.size -= col_size

    cpdef add_sv(self, int i):
        if self.verbose >= 2:
            print "Add SV", i

        cdef list[int].iterator it
        if self.support_vector[i] == -1:
            self.support_set.push_back(i)
            it = self.support_set.end()
            dec(it)
            self.support_it[i] = it
            self.support_vector[i] = self.support_set.size() - 1

    cpdef remove_sv(self, int i):
        if self.verbose >= 2:
            print "Remove SV", i

        cdef list[int].iterator it
        cdef map[int, double*].iterator it2
        cdef double* cache
        cdef int j

        if self.support_vector[i] >= 0:
            it = self.support_it[i]
            self.support_set.erase(it)
            k = self.support_vector[i]
            self.support_vector[i] = -1

            it2 = self.columns.begin()
            while it2 != self.columns.end():
                j = deref(it2).first
                self.columns[0][j][i] = 0
                inc(it2)

    cpdef int n_sv(self):
        return self.support_set.size()

    cpdef get_size(self):
        return self.size


def get_kernel(kernel, **kw):
    if kernel == "linear":
        return LinearKernel()
    elif kernel == "rbf":
        return RbfKernel(gamma=kw["gamma"])
    elif kernel == "poly" or kernel == "polynomial":
        return PolynomialKernel(degree=kw["degree"],
                                coef0=kw["coef0"],
                                gamma=kw["gamma"])
    elif kernel == "precomputed":
        return PrecomputedKernel()
