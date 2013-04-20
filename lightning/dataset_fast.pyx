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


DEF LINEAR_KERNEL = 1
DEF POLY_KERNEL = 2
DEF RBF_KERNEL = 3

KERNELS = {"linear" : LINEAR_KERNEL,
           "poly" : POLY_KERNEL,
           "polynomial" : POLY_KERNEL,
           "rbf" : RBF_KERNEL}


cdef double _linear_kernel(double* data_X,
                           double* data_Y,
                           int n_features):
    cdef int k
    cdef double dot = 0

    for k in xrange(n_features):
        dot += data_X[k] * data_Y[k]

    return dot


cdef double _poly_kernel(double* data_X,
                         double* data_Y,
                         int n_features,
                         double gamma,
                         double coef0,
                         int degree):
    cdef double dot = _linear_kernel(data_X, data_Y, n_features)
    return powi(coef0 + dot * gamma, degree)


cdef double _rbf_kernel(double* data_X,
                        double* data_Y,
                        int n_features,
                        double gamma):
    cdef double diff, value = 0

    for k in xrange(n_features):
        diff = data_X[k] - data_Y[k]
        value += diff * diff

    return exp(-gamma * value)


cdef double _kernel(double* data_X,
                    double* data_Y,
                    int n_features,
                    double gamma,
                    double coef0,
                    int degree,
                    int kernel):
    cdef ret = 0
    if kernel == LINEAR_KERNEL:
        ret = _linear_kernel(data_X, data_Y, n_features)
    elif kernel == POLY_KERNEL:
        ret = _poly_kernel(data_X, data_Y, n_features, gamma, coef0, degree)
    elif kernel == RBF_KERNEL:
        ret = _rbf_kernel(data_X, data_Y, n_features, gamma)
    return ret


cdef class KernelDataset(Dataset):

    def __init__(self,
                 np.ndarray[double, ndim=2, mode='c'] X,
                 np.ndarray[double, ndim=2, mode='c'] Y,
                 kernel="linear",
                 double gamma=0.1,
                 double coef0=1.0,
                 int degree=4,
                 long capacity=500,
                 int mb=1,
                 int verbose=0):

        # Input data
        self.n_samples = X.shape[0]
        self.n_features = Y.shape[0]
        self.data = <double*> X.data
        self.n_features_Y = Y.shape[1]
        self.data_Y = <double*> Y.data

        # Kernel parameters
        self.kernel = KERNELS[kernel]
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree

        # Cache
        if mb:
            self.capacity = capacity * (1 << 20)
        else:
            self.capacity = capacity
        self.verbose = verbose
        self.size = 0

    def __cinit__(self,
                  np.ndarray[double, ndim=2, mode='c'] X,
                  np.ndarray[double, ndim=2, mode='c'] Y,
                  kernel="linear",
                  double gamma=0.1,
                  double coef0=1.0,
                  int degree=4,
                  long capacity=500,
                  int mb=1,
                  int verbose=0):
        cdef int i
        cdef int n_samples = X.shape[0]

        # Allocate indices.
        self.indices = <int*> stdlib.malloc(sizeof(int) * n_samples)
        for i in xrange(n_samples):
            self.indices[i] = i

        # Allocate support set.
        self.support_set = new list[int]()
        self.support_vector = <int*> stdlib.malloc(sizeof(int) * n_samples)
        self.support_it = <list[int].iterator*> \
            stdlib.malloc(sizeof(list[int].iterator) * n_samples)

        # Allocate containers for cache.
        self.n_computed = <int*> stdlib.malloc(sizeof(int) * n_samples)
        self.columns = new map[int, double*]()

        for i in xrange(n_samples):
            self.support_vector[i] = -1
            self.n_computed[i] = 0

        self.cache = <double*> stdlib.malloc(sizeof(double) * n_samples)

    def __dealloc__(self):
        # De-allocate indices.
        stdlib.free(self.indices)

        # De-allocate support set.
        del self.support_set
        stdlib.free(self.support_vector)
        stdlib.free(self.support_it)

        # De-allocate containers for cache.
        self._clear_columns(self.n_samples)
        self.columns.clear()
        del self.columns
        stdlib.free(self.n_computed)

        stdlib.free(self.cache)

    cdef void _kernel_column(self, int j, double *out):
        cdef double* data_X
        cdef double* data_Y

        data_X = self.data
        data_Y = self.data_Y + j * self.n_features_Y

        for i in xrange(self.n_samples):
            out[i] = _kernel(data_X, data_Y, self.n_features_Y,
                             self.gamma, self.coef0, self.degree, self.kernel)
            data_X += self.n_features_Y

    cdef void _kernel_column_sv(self, int j, double *out):
        cdef double* data_X
        cdef double* data_Y
        cdef list[int].iterator it
        cdef int s

        data_Y = self.data_Y + j * self.n_features_Y

        it = self.support_set.begin()
        while it != self.support_set.end():
            s = deref(it)
            data_X = self.data + s * self.n_features_Y

            if out[s] == 0:
                out[s] = _kernel(data_X, data_Y, self.n_features_Y,
                                 self.gamma, self.coef0, self.degree,
                                 self.kernel)
            inc(it)

    cdef void _create_column(self, int i):
        cdef int n_computed = self.n_computed[i]

        if n_computed != 0:
            return

        cdef int col_size = self.n_samples * sizeof(double)

        if self.size + col_size > self.capacity:
            if self.verbose >= 2:
                print "Empty cache by half"
            self._clear_columns(self.columns.size() / 2)

        self.columns[0][i] = <double*> stdlib.calloc(self.n_samples,
                                                     sizeof(double))
        self.size += col_size

    cdef void _clear_columns(self, int n):
        cdef map[int, double*].iterator it
        it = self.columns.begin()
        cdef int i = 0
        cdef int col_size

        while it != self.columns.end():
            col_size = self.n_samples * sizeof(double)
            stdlib.free(deref(it).second)
            self.n_computed[deref(it).first] = 0
            self.size -= col_size
            self.columns.erase(postinc(it))

            if i >= n - 1:
                break

            i += 1

    cdef double* _get_column(self, int j):
        cdef int i = 0

        if self.capacity == 0:
            self._kernel_column(j, self.cache)
            return self.cache

        cdef int n_computed = self.n_computed[j]

        self._create_column(j)

        cdef double* cache = &(self.columns[0][j][0])

        if n_computed == -1:
            # Full column is already computed.
            return cache
        else:
            # All elements must be computed.
            self._kernel_column(j, cache)

        self.n_computed[j] = -1

        return cache

    cdef void get_column_ptr(self,
                             int j,
                             int** indices,
                             double** data,
                             int* n_nz):

        indices[0] = self.indices
        data[0] = self._get_column(j)
        n_nz[0] = self.n_samples

    cdef double* get_column_sv_ptr(self, int j):

        cdef int s
        cdef int n_computed = self.n_computed[j]
        cdef int ssize = self.support_set.size()

        if ssize == 0:
            return self.cache

        self._create_column(j)
        cdef double* cache = &(self.columns[0][j][0])

        if n_computed == -1:
            # Full column is already computed.
            return cache

        self._kernel_column_sv(j, cache)
        self.n_computed[j] = ssize

        return cache

    cdef void get_column_sv_out(self, int j, double* out):
        cdef double* cache
        cdef int s
        cdef list[int].iterator it

        cache = self.get_column_sv_ptr(j)

        it = self.support_set.begin()
        while it != self.support_set.end():
            s = deref(it)
            out[s] = cache[s]
            inc(it)

    cpdef get_column_sv(self, int j):
        cdef np.ndarray[double, ndim=1, mode='c'] out
        out = np.zeros(self.n_samples, dtype=np.float64)
        self.get_column_sv_out(j, <double*>out.data)
        return out

    cdef void get_diag_out(self, double* out):
        cdef int i
        cdef double* data_X = self.data

        for i in xrange(self.n_samples):
            if self.kernel == RBF_KERNEL:
                out[i] = 1
            else:
                out[i] = _kernel(data_X, data_X, self.n_features_Y,
                                 self.gamma, self.coef0, self.degree,
                                 self.kernel)
            data_X += self.n_features_Y

    cpdef get_diag(self):
        cdef np.ndarray[double, ndim=1, mode='c'] out
        out = np.zeros(self.n_samples, dtype=np.float64)
        self.get_diag_out(<double*>out.data)
        return out

    cpdef remove_column(self, int i):
        if self.verbose >= 2:
            print "Remove column SV", i

        cdef map[int, double*].iterator it
        cdef int col_size = self.n_samples * sizeof(double)

        it = self.columns.find(i)

        if it != self.columns.end():
            self.n_computed[deref(it).first] = 0
            stdlib.free(deref(it).second)
            self.columns.erase(it)
            self.size -= col_size

    cpdef add_sv(self, int i):
        if self.verbose >= 2:
            print "Add SV", i

        cdef list[int].iterator it
        if self.support_vector[i] == -1:
            self.support_set.push_back(i)
            it = self.support_set.end()
            dec(it)
            self.support_it[i] = it
            self.support_vector[i] = self.support_set.size() - 1

    cpdef remove_sv(self, int i):
        if self.verbose >= 2:
            print "Remove SV", i

        cdef list[int].iterator it
        cdef map[int, double*].iterator it2
        cdef double* cache
        cdef int j

        if self.support_vector[i] >= 0:
            it = self.support_it[i]
            self.support_set.erase(it)
            k = self.support_vector[i]
            self.support_vector[i] = -1

            it2 = self.columns.begin()
            while it2 != self.columns.end():
                j = deref(it2).first
                self.columns[0][j][i] = 0
                inc(it2)

    cpdef int n_sv(self):
        return self.support_set.size()

    cpdef get_size(self):
        return self.size

    cpdef double get_element(self, int i, int j):
        cdef double* data_X
        cdef double* data_Y

        if i == j and self.kernel == RBF_KERNEL:
            return 1.0
        else:
            data_X = self.data + i * self.n_features_Y
            data_Y = self.data_Y + j * self.n_features_Y
            return _kernel(data_X, data_Y, self.n_features_Y,
                           self.gamma, self.coef0, self.degree,
                           self.kernel)

    def dot(self, coef):
        cdef int n_features = coef.shape[0]
        cdef int n_vectors = coef.shape[1]

        cdef np.ndarray[double, ndim=2, mode='c'] out
        out = np.zeros((self.n_samples, n_vectors), dtype=np.float64)

        cdef np.ndarray[double, ndim=2, mode='c'] coef_
        coef_ = np.ascontiguousarray(coef, dtype=np.float64)

        cdef int i, j, k

        for j in xrange(n_features):
            self._kernel_column(j, self.cache)

            for i in xrange(self.n_samples):
                for k in xrange(n_vectors):
                    out[i, k] += self.cache[i] * coef_[j, k]

        return out

