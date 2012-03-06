# Author: Mathieu Blondel
# License: BSD

cimport numpy as np

cdef class Kernel:

    cpdef double compute(self,
                         np.ndarray[double, ndim=2, mode='c'] X,
                         int i,
                         np.ndarray[double, ndim=2, mode='c'] Y,
                         int j)


cdef class LinearKernel(Kernel):

    cpdef double compute(self,
                         np.ndarray[double, ndim=2, mode='c'] X,
                         int i,
                         np.ndarray[double, ndim=2, mode='c'] Y,
                         int j)


cdef class PolynomialKernel(Kernel):
    cdef int degree
    cdef double coef0, gamma

    cpdef double compute(self,
                         np.ndarray[double, ndim=2, mode='c'] X,
                         int i,
                         np.ndarray[double, ndim=2, mode='c'] Y,
                         int j)


cdef class RbfKernel(Kernel):
    cdef double gamma

    cpdef double compute(self,
                         np.ndarray[double, ndim=2, mode='c'] X,
                         int i,
                         np.ndarray[double, ndim=2, mode='c'] Y,
                         int j)


cdef class PrecomputedKernel(Kernel):

    cpdef double compute(self,
                         np.ndarray[double, ndim=2, mode='c'] X,
                         int i,
                         np.ndarray[double, ndim=2, mode='c'] Y,
                         int j)
