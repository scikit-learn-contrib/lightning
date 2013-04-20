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
