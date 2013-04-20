# Author: Mathieu Blondel
# License: BSD

from libcpp.list cimport list
from libcpp.map cimport map

cdef class Dataset:

    cdef int n_samples
    cdef int n_features

    cdef void get_column_ptr(self,
                             int j,
                             int** indices,
                             double** data,
                             int* n_nz)

    cpdef get_column(self, int j)

    cdef void get_row_ptr(self,
                             int i,
                             int** indices,
                             double** data,
                             int* n_nz)

    cpdef get_row(self, int i)

    cpdef int get_n_samples(self)
    cpdef int get_n_features(self)


cdef class ContiguousDataset(Dataset):

    cdef int* indices
    cdef double* data
    cdef object X

    cdef void get_row_ptr(self,
                          int i,
                          int** indices,
                          double** data,
                          int* n_nz)


cdef class FortranDataset(Dataset):

    cdef int* indices
    cdef double* data
    cdef object X

    cdef void get_column_ptr(self,
                             int j,
                             int** indices,
                             double** data,
                             int* n_nz)


cdef class CSRDataset(Dataset):

    cdef int* indices
    cdef double* data
    cdef int* indptr
    cdef object X

    cdef void get_row_ptr(self,
                          int i,
                          int** indices,
                          double** data,
                          int* n_nz)


cdef class CSCDataset(Dataset):

    cdef int* indices
    cdef double* data
    cdef int* indptr
    cdef object X

    cdef void get_column_ptr(self,
                             int j,
                             int** indices,
                             double** data,
                             int* n_nz)


cdef class KernelDataset(Dataset):

    # Input data
    cdef int* indices
    cdef double* data
    cdef int n_features_Y
    cdef double* data_Y

    # Kernel parameters
    cdef int kernel
    cdef double coef0
    cdef double gamma
    cdef int degree

    # Cache
    cdef double* cache
    cdef map[int, double*]* columns
    cdef int* n_computed
    cdef long capacity
    cdef int verbose
    cdef long size

    # Support set
    cdef list[int]* support_set
    cdef list[int].iterator* support_it
    cdef int* support_vector

    # Methods
    cdef void _kernel_column(self, int j, double *out)
    cdef void _kernel_column_sv(self, int j, double *out)

    cdef void _create_column(self, int i)
    cdef void _clear_columns(self, int n)
    cdef double* _get_column(self, int j)
    cdef void get_column_ptr(self,
                             int j,
                             int** indices,
                             double** data,
                             int* n_nz)
    cdef double* get_column_sv_ptr(self, int j)
    cdef void get_column_sv_out(self, int j, double* out)
    cpdef get_column_sv(self, int j)

    cdef void get_diag_out(self, double*out)
    cpdef get_diag(self)

    cpdef double get_element(self, int i, int j)

    cpdef remove_column(self, int i)
    cpdef add_sv(self, int i)
    cpdef remove_sv(self, int i)
    cpdef int n_sv(self)
    cpdef get_size(self)
