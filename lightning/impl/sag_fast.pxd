# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: language_level=3
#
# Authors: Mathieu Blondel
#          Fabian Pedregosa
#          Arnaud Rachez
# License: BSD

cimport numpy as np

cdef class Penalty:

    # wether it supports lagged updates. 0 indicates no support,
    # and anything else indicates support.
    # In the case of no support, only the method
    # projection will be used and not projection_lagged
    cdef bint support_lagged

    cdef void projection(self,
                         double* w,
                         int* indices,
                         double stepsize,
                         int n_nz)

    cdef void projection_lagged(self,
                                int t,
                                double* w,
                                double* g_sum,
                                int* indices,
                                double stepsize_prox,
                                double stepsize_grad,
                                double* lag_scaling,
                                int n_nz,
                                int* last,
                                double* scaling_seq)

    cdef double regularization(self, np.ndarray[double, ndim=1]coef)
