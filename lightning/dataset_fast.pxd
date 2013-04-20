# Author: Mathieu Blondel
# License: BSD

cdef class Dataset:

    cdef int n_samples
    cdef int n_features

    cpdef int get_n_samples(self)
    cpdef int get_n_features(self)


cdef class RowDataset(Dataset):

    cdef void get_row_ptr(self,
                             int i,
                             int** indices,
                             double** data,
                             int* n_nz)

    cpdef get_row(self, int i)


cdef class ColumnDataset(Dataset):

    cdef void get_column_ptr(self,
                             int j,
                             int** indices,
                             double** data,
                             int* n_nz)

    cpdef get_column(self, int j)


cdef class ContiguousDataset(RowDataset):

    cdef int* indices
    cdef double* data
    cdef object X

    cdef void get_row_ptr(self,
                          int i,
                          int** indices,
                          double** data,
                          int* n_nz)


cdef class FortranDataset(ColumnDataset):

    cdef int* indices
    cdef double* data
    cdef object X

    cdef void get_column_ptr(self,
                             int j,
                             int** indices,
                             double** data,
                             int* n_nz)


cdef class CSRDataset(RowDataset):

    cdef int* indices
    cdef double* data
    cdef int* indptr
    cdef object X

    cdef void get_row_ptr(self,
                          int i,
                          int** indices,
                          double** data,
                          int* n_nz)


cdef class CSCDataset(ColumnDataset):

    cdef int* indices
    cdef double* data
    cdef int* indptr
    cdef object X

    cdef void get_column_ptr(self,
                             int j,
                             int** indices,
                             double** data,
                             int* n_nz)
