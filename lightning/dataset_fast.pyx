# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
#
# Author: Mathieu Blondel
# License: BSD

from cython.operator cimport dereference as deref
from cython.operator cimport preincrement as inc
from cython.operator cimport postincrement as postinc
from cython.operator cimport predecrement as dec

from libcpp.list cimport list
from libcpp.map cimport map
from libc cimport stdlib

import numpy as np
cimport numpy as np
np.import_array()

import scipy.sparse as sp

from sklearn.utils.extmath import safe_sparse_dot

cdef extern from "math.h":
   double exp(double)

cdef double powi(double base, int times):
    cdef double tmp = base, ret = 1.0

    cdef int t = times

    while t > 0:
        if t % 2 == 1:
            ret *= tmp
        tmp = tmp * tmp

        t /= 2

    return ret

cdef class Dataset:

    cdef void get_column_ptr(self,
                             int j,
                             int** indices,
                             double** data,
                             int* n_nz):
        raise NotImplementedError()

    cpdef get_column(self, int j):
        cdef double* data
        cdef int* indices
        cdef int n_nz
        cdef np.npy_intp shape[1]

        self.get_column_ptr(j, &indices, &data, &n_nz)

        shape[0] = <np.npy_intp> self.n_samples
        indices_ = np.PyArray_SimpleNewFromData(1, shape, np.NPY_INT, indices)
        data_ = np.PyArray_SimpleNewFromData(1, shape, np.NPY_DOUBLE, data)

        return indices_, data_, n_nz

    cdef void get_row_ptr(self,
                          int i,
                          int** indices,
                          double** data,
                          int* n_nz):
        raise NotImplementedError()

    cpdef get_row(self, int i):
        cdef double* data
        cdef int* indices
        cdef int n_nz
        cdef np.npy_intp shape[1]

        self.get_row_ptr(i, &indices, &data, &n_nz)

        shape[0] = <np.npy_intp> self.n_features
        indices_ = np.PyArray_SimpleNewFromData(1, shape, np.NPY_INT, indices)
        data_ = np.PyArray_SimpleNewFromData(1, shape, np.NPY_DOUBLE, data)

        return indices_, data_, n_nz

    cpdef int get_n_samples(self):
        return self.n_samples

    cpdef int get_n_features(self):
        return self.n_features

    def dot(self, coef):
        return NotImplementedError()


cdef class ContiguousDataset(Dataset):

    def __init__(self, np.ndarray[double, ndim=2, mode='c'] X):
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        self.data = <double*> X.data
        self.X = X

    def __cinit__(self, np.ndarray[double, ndim=2, mode='c'] X):
        cdef int i
        cdef int n_features = X.shape[1]
        self.indices = <int*> stdlib.malloc(sizeof(int) * n_features)
        for j in xrange(n_features):
            self.indices[j] = j

    def __dealloc__(self):
        stdlib.free(self.indices)

    cdef void get_row_ptr(self,
                          int i,
                          int** indices,
                          double** data,
                          int* n_nz):
        indices[0] = self.indices
        data[0] = self.data + i * self.n_features
        n_nz[0] = self.n_features

    def dot(self, coef):
        return safe_sparse_dot(self.X, coef)


cdef class FortranDataset(Dataset):

    def __init__(self, np.ndarray[double, ndim=2, mode='fortran'] X):
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        self.data = <double*> X.data
        self.X = X

    def __cinit__(self, np.ndarray[double, ndim=2, mode='fortran'] X):
        cdef int i
        cdef int n_samples = X.shape[0]
        self.indices = <int*> stdlib.malloc(sizeof(int) * n_samples)
        for i in xrange(n_samples):
            self.indices[i] = i

    def __dealloc__(self):
        stdlib.free(self.indices)

    cdef void get_column_ptr(self,
                             int j,
                             int** indices,
                             double** data,
                             int* n_nz):
        indices[0] = self.indices
        data[0] = self.data + j * self.n_samples
        n_nz[0] = self.n_samples

    def dot(self, coef):
        return safe_sparse_dot(self.X, coef)


cdef class CSRDataset(Dataset):

    def __init__(self, X):
        cdef np.ndarray[double, ndim=1, mode='c'] X_data = X.data
        cdef np.ndarray[int, ndim=1, mode='c'] X_indices = X.indices
        cdef np.ndarray[int, ndim=1, mode='c'] X_indptr = X.indptr

        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        self.data = <double*> X_data.data
        self.indices = <int*> X_indices.data
        self.indptr = <int*> X_indptr.data

        self.X = X

    cdef void get_row_ptr(self,
                          int i,
                          int** indices,
                          double** data,
                          int* n_nz):
        indices[0] = self.indices + self.indptr[i]
        data[0] = self.data + self.indptr[i]
        n_nz[0] = self.indptr[i + 1] - self.indptr[i]

    def dot(self, coef):
        return safe_sparse_dot(self.X, coef)


cdef class CSCDataset(Dataset):

    def __init__(self, X):
        cdef np.ndarray[double, ndim=1, mode='c'] X_data = X.data
        cdef np.ndarray[int, ndim=1, mode='c'] X_indices = X.indices
        cdef np.ndarray[int, ndim=1, mode='c'] X_indptr = X.indptr

        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        self.data = <double*> X_data.data
        self.indices = <int*> X_indices.data
        self.indptr = <int*> X_indptr.data

        self.X = X

    cdef void get_column_ptr(self,
                             int j,
                             int** indices,
                             double** data,
                             int* n_nz):
        indices[0] = self.indices + self.indptr[j]
        data[0] = self.data + self.indptr[j]
        n_nz[0] = self.indptr[j + 1] - self.indptr[j]

    def dot(self, coef):
        return safe_sparse_dot(self.X, coef)


def get_dataset(X, order="c"):
    if isinstance(X, Dataset):
        return X

    if sp.isspmatrix(X):
        if order == "fortran":
            X = X.tocsc()
            ds = CSCDataset(X)
        else:
            X = X.tocsr()
            ds = CSRDataset(X)
    else:
        if order == "fortran":
            X = np.asfortranarray(X, dtype=np.float64)
            ds = FortranDataset(X)
        else:
            X = np.ascontiguousarray(X, dtype=np.float64)
            ds = ContiguousDataset(X)
    return ds
