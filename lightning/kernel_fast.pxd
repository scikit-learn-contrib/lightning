# Author: Mathieu Blondel
# License: BSD

from libcpp.list cimport list as list
from libcpp.map cimport map

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


cdef class KernelCache:
    cdef Kernel kernel
    cdef long n_samples
    cdef list[long]* support_set
    cdef list[long].iterator* support_it
    cdef int* is_support_vector
    cdef map[long, double*]* columns
    cdef long* n_computed
    cdef int capacity
    cdef int size

    cdef _create_column(self, long i)
    cdef _clear_columns(self, long n)

    cpdef double compute(self,
                         np.ndarray[double, ndim=2, mode='c'] X,
                         int i,
                         np.ndarray[double, ndim=2, mode='c'] Y,
                         int j)

    cpdef compute_column(self,
                         np.ndarray[double, ndim=2, mode='c'] X,
                         np.ndarray[double, ndim=2, mode='c'] Y,
                         long j,
                         np.ndarray[double, ndim=1, mode='c'] out)

    cpdef add_sv(self, long i)

    cpdef remove_sv(self, long i)

    cpdef long n_sv(self)

    cpdef get_size(self)
