
cimport numpy as np

cdef class Penalty:

    cdef void projection(self,
                         double* w,
                         int* indices,
                         double stepsize,
                         double w_scale,
                         int n_nz)
    
    cdef void projection_lagged(self,
                                double* w,
                                int* indices,
                                double stepsize,
                                double* scale_cumm,
                                int t,
                                int n_nz,
                                int* last)

    cdef double regularization(self, np.ndarray[double, ndim=1]coef)
