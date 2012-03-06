# Author: Mathieu Blondel
# License: BSD

cimport numpy as np

cdef class Kernel:

    cpdef double compute(self,
                         np.ndarray[double, ndim=2, mode='c'] X,
                         int i,
                         np.ndarray[double, ndim=2, mode='c'] Y,
                         int j)

    cpdef double compute_self(self,
                              np.ndarray[double, ndim=2, mode='c'] X,
                              int i)

    cpdef compute_diag(self,
                       np.ndarray[double, ndim=2, mode='c'] X,
                       np.ndarray[double, ndim=1, mode='c'] out)

    cdef void compute_diag_ptr(self,
                               np.ndarray[double, ndim=2, mode='c'] X,
                               double* out)

    cpdef compute_column(self,
                         np.ndarray[double, ndim=2, mode='c'] X,
                         np.ndarray[double, ndim=2, mode='c'] Y,
                         int j,
                         np.ndarray[double, ndim=1, mode='c'] out)

    cdef void compute_column_ptr(self,
                                 np.ndarray[double, ndim=2, mode='c'] X,
                                 np.ndarray[double, ndim=2, mode='c'] Y,
                                 int j,
                                 double* out)


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

    cpdef double compute_self(self,
                              np.ndarray[double, ndim=2, mode='c'] X,
                              int i)


cdef class PrecomputedKernel(Kernel):

    cpdef double compute(self,
                         np.ndarray[double, ndim=2, mode='c'] X,
                         int i,
                         np.ndarray[double, ndim=2, mode='c'] Y,
                         int j)
