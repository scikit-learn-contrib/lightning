from libcpp.list cimport list as list

cimport numpy as np

from lightning.kernel_fast cimport Kernel

cdef int get_select_method(selection)

cdef int select_sv(np.ndarray[long, ndim=1, mode='c'] A,
                   int start,
                   int search_size,
                   int max_size,
                   int select_method,
                   np.ndarray[double, ndim=1, mode='c'] alpha,
                   double b,
                   np.ndarray[double, ndim=2, mode='c'] X,
                   np.ndarray[double, ndim=1] y,
                   Kernel kernel,
                   list[long]& support_set,
                   np.ndarray[long, ndim=1, mode='c'] support_vectors)
