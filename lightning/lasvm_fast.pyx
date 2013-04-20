# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
#
# Author: Mathieu Blondel
# License: BSD

import sys

from cython.operator cimport dereference as deref, preincrement as inc
from libcpp.list cimport list

import numpy as np
cimport numpy as np

from lightning.dataset_fast cimport KernelDataset
from lightning.select_fast cimport get_select_method, select_sv
from lightning.random.random_fast cimport RandomState


cdef extern from "float.h":
   double DBL_MAX

cdef struct Intpair:
    int left
    int right


cdef Intpair _argmin_argmax(double* y,
                            double* g,
                            list[int]* support_set,
                            double* alpha,
                            double C):

    cdef int s, sel_min, sel_max
    cdef double min_ = DBL_MAX
    cdef double max_ = -DBL_MAX
    cdef double As
    cdef double Bs
    cdef double Cy

    cdef list[int].iterator it = support_set.begin()
    while it != support_set.end():
        s = deref(it)
        Cy = C * y[s]

        As = min(0, Cy)
        Bs = max(0, Cy)

        if g[s] > max_ and alpha[s] < Bs:
            max_ = g[s]
            sel_max = s

        if g[s] < min_ and alpha[s] > As:
            min_ = g[s]
            sel_min = s

        inc(it)

    cdef Intpair ret
    ret.left = sel_min
    ret.right = sel_max

    return ret


cdef void _update(KernelDataset kds,
                  double *y,
                  double *g,
                  double *alpha,
                  int i,
                  int j,
                  double Aj,
                  double Bi,
                  double* col,
                  double* col2):
    cdef list[int]* support_set = kds.support_set
    cdef double Kii, Kjj, Kij, Kis, Kjs

    # Need only three elements.
    Kii = kds.get_element(i, i)
    Kjj = kds.get_element(j, j)
    Kij = kds.get_element(i, j)
    cdef double lambda_ = min((g[i] - g[j]) / (Kii + Kjj - 2 * Kij),
                              min(Bi - alpha[i], alpha[j] - Aj))
    alpha[i] += lambda_
    alpha[j] -= lambda_


    cdef int s
    cdef list[int].iterator it = support_set.begin()
    kds.get_column_sv_out(i, col)
    kds.get_column_sv_out(j, col2)
    while it != support_set.end():
        s = deref(it)
        g[s] -= lambda_ * (col[s] - col2[s])
        inc(it)


cdef void _process(int k,
                   KernelDataset kds,
                   double* y,
                   double* alpha,
                   double* g,
                   double C,
                   double tau,
                   double* col,
                   double* col2):

    cdef int* support_vectors = kds.support_vector
    cdef list[int]* support_set = kds.support_set

    if support_vectors[k] >= 0:
        return

    alpha[k] = 0

    cdef int s, j, i
    cdef double pred = 0

    cdef list[int].iterator it = support_set.begin()
    kds.get_column_sv_out(k, col)
    while it != support_set.end():
        s = deref(it)
        pred += alpha[s] * col[s]
        inc(it)

    g[k] = y[k] - pred

    kds.add_sv(k)

    if y[k] == 1:
        i = k
        j = _argmin_argmax(y, g, support_set, alpha, C).left
    else:
        j = k
        i = _argmin_argmax(y, g, support_set, alpha, C).right

    cdef double Aj = min(0, C * y[j])
    cdef double Bi = max(0, C * y[i])

    cdef int violating_pair
    violating_pair = alpha[i] < Bi and alpha[j] > Aj and g[i] - g[j] > tau

    if not violating_pair:
        return

    _update(kds, y, g, alpha, i, j, Aj, Bi, col, col2)



cdef void _reprocess(KernelDataset kds,
                     double* y,
                     double* alpha,
                     double* g,
                     double* b,
                     double* delta,
                     double C,
                     double tau,
                     double* col,
                     double* col2):

    cdef int* support_vectors = kds.support_vector
    cdef list[int]* support_set = kds.support_set

    cdef Intpair p
    p = _argmin_argmax(y, g, support_set, alpha, C)
    cdef int i = p.right
    cdef int j = p.left

    cdef double Aj = min(0, C * y[j])
    cdef double Bi = max(0, C * y[i])

    cdef int violating_pair
    violating_pair = alpha[i] < Bi and alpha[j] > Aj and g[i] - g[j] > tau

    if not violating_pair:
        return

    _update(kds, y, g, alpha, i, j, Aj, Bi, col, col2)

    p = _argmin_argmax(y, g, support_set, alpha, C)
    i = p.right
    j = p.left

    cdef int s, k = 0
    cdef int n_removed = 0

    cdef list[int].iterator it = support_set.begin()
    cdef list[int] to_remove
    while it != support_set.end():
        s = deref(it)

        if alpha[s] == 0:
            if (y[s] == -1 and g[s] >= g[i]) or (y[s] == 1 and g[s] <= g[j]):
                to_remove.push_back(s)

        inc(it)

    it = to_remove.begin()
    while it != to_remove.begin():
        kds.remove_sv(deref(it))
        inc(it)

    b[0] = (g[i] + g[j]) / 2
    delta[0] = g[i] - g[j]


cdef void _boostrap(index,
                    KernelDataset kds,
                    double* y,
                    double* alpha,
                    double* g,
                    double* col,
                    rs):
    cdef int* support_vectors = kds.support_vector
    cdef list[int]* support_set = kds.support_set
    cdef np.ndarray[int, ndim=1, mode='c'] A = index
    cdef int n_pos = 0
    cdef int n_neg = 0
    cdef int i, s
    cdef Py_ssize_t n_samples = index.shape[0]
    cdef int* indices
    cdef int n_nz

    rs.shuffle(A)

    for i in xrange(n_samples):
        s = A[i]

        if (y[s] == -1 and n_neg < 5) or (y[s] == 1 and n_pos < 5):
            kds.add_sv(s)

            alpha[s] = y[s]
            # Entire sth column
            kds.get_column_ptr(s, &indices, &col, &n_nz)

            for j in xrange(n_samples):
                # All elements of the s-th column.
                g[j] -= y[s] * col[j]

            if y[s] == -1:
                n_neg += 1
            else:
                n_pos += 1

        if (n_neg == 5 and n_pos == 5):
            break


cdef void _boostrap_warm_start(index,
                               KernelDataset kds,
                               double* y,
                               double* alpha,
                               double* g,
                               double* col):
    cdef int* support_vectors = kds.support_vector
    cdef list[int]* support_set = kds.support_set
    cdef np.ndarray[int, ndim=1, mode='c'] A = index
    cdef int i, s
    cdef Py_ssize_t n_samples = index.shape[0]
    cdef int* indices
    cdef int n_nz


    for i in xrange(n_samples):
        if alpha[i] != 0:
            kds.get_column_ptr(i, &indices, &col, &n_nz)

            for j in xrange(n_samples):
                # All elements of the i-th column.
                g[j] -= alpha[i] * col[j]

            kds.add_sv(i)


def _lasvm(self,
           np.ndarray[double, ndim=1, mode='c'] alpha,
           KernelDataset kds,
           np.ndarray[double, ndim=1] y,
           selection,
           int search_size,
           int permute,
           termination,
           int n_components,
           double tau,
           int finish_step,
           double C,
           int max_iter,
           RandomState rs,
           callback,
           int verbose,
           int warm_start):

    cdef Py_ssize_t n_samples = kds.get_n_samples()

    # Active set.
    cdef np.ndarray[int, ndim=1, mode='c'] A
    A = np.arange(n_samples, dtype=np.int32)

    # Initialization
    cdef np.ndarray[double, ndim=1, mode='c'] g
    cdef np.ndarray[double, ndim=1, mode='c'] b
    cdef np.ndarray[double, ndim=1, mode='c'] delta
    cdef np.ndarray[double, ndim=1, mode='c'] col
    cdef np.ndarray[double, ndim=1, mode='c'] col2

    g = y.copy()
    b = np.zeros(1, dtype=np.float64)
    delta = np.zeros(1, dtype=np.float64)
    col = np.zeros(n_samples, dtype=np.float64)
    col2 = np.zeros(n_samples, dtype=np.float64)

    cdef int it, i, j, s, k
    cdef int n_pos, n_neg
    cdef int stop = 0
    cdef list[int] support_set

    cdef int check_n_sv = termination == "n_components"
    cdef int has_callback = callback is not None
    cdef int select_method = get_select_method(selection)

    # Data pointers
    cdef double* y_ptr = <double*>y.data
    cdef double* alpha_ptr = <double*>alpha.data
    cdef double* g_ptr = <double*>g.data
    cdef double* b_ptr = <double*>b.data
    cdef double* delta_ptr = <double*>delta.data
    cdef double* col_ptr = <double*>col.data
    cdef double* col2_ptr = <double*>col2.data

    # Boostrap model.
    if warm_start:
        _boostrap_warm_start(A, kds, y_ptr, alpha_ptr, g_ptr, col_ptr)
    else:
        _boostrap(A, kds, y_ptr, alpha_ptr, g_ptr, col_ptr, rs)

    for it in xrange(max_iter):

        if verbose >= 1:
            print "\nIteration", it

        if permute:
            rs.shuffle(A)

        for i in xrange(n_samples):
            # Select a support vector candidate.
            s = select_sv(A, search_size, n_samples, select_method,
                          alpha, b[0], kds, y, 1, rs)

            # Attempt to add it.
            _process(s, kds, y_ptr, alpha_ptr, g_ptr, C, tau, col_ptr, col2_ptr)

            # Remove blatant non support vectors.
            _reprocess(kds, y_ptr, alpha_ptr, g_ptr, b_ptr, delta_ptr, C, tau,
                       col_ptr, col2_ptr)

            # Exit if necessary.
            if check_n_sv and kds.n_sv() >= n_components:
                stop = 1
                break

            # Callback
            if has_callback and i % 100 == 0:
                ret = callback(self)
                if ret is not None:
                    stop = 1
                    break

            # Output progress.
            if verbose >= 1 and i % 100 == 0:
                sys.stdout.write(".")
                sys.stdout.flush()

        # end for

        if stop:
            break

    if finish_step:
        while delta[0] > tau:
            _reprocess(kds, y_ptr, alpha_ptr, g_ptr, b_ptr, delta_ptr, C, tau,
                       col_ptr, col2_ptr)

    if verbose >= 1:
        print

    # Return intercept.
    return b[0]
