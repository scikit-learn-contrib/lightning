from libcpp.list cimport list

cimport numpy as np

from lightning.kernel_fast cimport KernelCache
from lightning.random.random_fast cimport RandomState

cdef int get_select_method(selection)

cdef int select_sv(np.ndarray[int, ndim=1, mode='c'] A,
                   int search_size,
                   int max_size,
                   int select_method,
                   np.ndarray[double, ndim=1, mode='c'] alpha,
                   double b,
                   np.ndarray[double, ndim=2, mode='c'] X,
                   np.ndarray[double, ndim=1] y,
                   KernelCache kcache,
                   np.ndarray[double, ndim=1, mode='c'] col,
                   int check_duplicates,
                   RandomState rs)

cdef int select_sv_precomputed(np.ndarray[int, ndim=1, mode='c'] A,
                               int search_size,
                               int max_size,
                               int select_method,
                               np.ndarray[double, ndim=1, mode='c'] errors,
                               KernelCache kcache,
                               int check_duplicates,
                               RandomState rs)

