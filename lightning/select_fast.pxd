from libcpp.list cimport list

cimport numpy as np

from lightning.random.random_fast cimport RandomState
from lightning.dataset_fast cimport KernelDataset

cdef int get_select_method(selection)

cdef int select_sv(np.ndarray[int, ndim=1, mode='c'] A,
                   int search_size,
                   int max_size,
                   int select_method,
                   np.ndarray[double, ndim=1, mode='c'] alpha,
                   double b,
                   KernelDataset kds,
                   np.ndarray[double, ndim=1] y,
                   int check_duplicates,
                   RandomState rs)

cdef int select_sv_precomputed(int* A,
                              int search_size,
                              int max_size,
                              int select_method,
                              double* errors,
                              RandomState rs)
