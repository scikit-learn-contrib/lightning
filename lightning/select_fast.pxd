from libcpp.list cimport list

cimport numpy as np

from lightning.kernel_fast cimport KernelCache

cdef int get_select_method(selection)

cdef int select_sv(np.ndarray[int, ndim=1, mode='c'] A,
                   int start,
                   int search_size,
                   int max_size,
                   int select_method,
                   np.ndarray[double, ndim=1, mode='c'] alpha,
                   double b,
                   np.ndarray[double, ndim=2, mode='c'] X,
                   np.ndarray[double, ndim=1] y,
                   KernelCache kcache,
                   np.ndarray[double, ndim=1, mode='c'] col)

cdef int select_sv_precomputed(np.ndarray[int, ndim=1, mode='c'] A,
                               int start,
                               int search_size,
                               int max_size,
                               int select_method,
                               np.ndarray[double, ndim=1, mode='c'] errors,
                               KernelCache kcache)

cdef int update_start(int start,
                      int select_method,
                      int search_size,
                      int active_size,
                      np.ndarray[int, ndim=1, mode='c'] index,
                      rs)
