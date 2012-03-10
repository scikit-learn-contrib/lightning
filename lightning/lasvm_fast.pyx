# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
#
# Author: Mathieu Blondel
# License: BSD

from cython.operator cimport dereference as deref, preincrement as inc
from libcpp.list cimport list as list

import numpy as np
cimport numpy as np

from lightning.kernel_fast cimport Kernel
from lightning.select_fast cimport get_select_method, select_sv, update_start

cdef extern from "float.h":
   double DBL_MAX


cdef int _argmax(np.ndarray[double, ndim=1] y,
                 np.ndarray[double, ndim=1, mode='c'] g,
                 list[long]& support_set,
                 np.ndarray[double, ndim=1, mode='c'] alpha,
                 double C):

    cdef int s, selected
    cdef double max_ = -DBL_MAX
    cdef double Bs

    cdef list[long].iterator it = support_set.begin()
    while it != support_set.end():
        s = deref(it)
        Bs = max(0, C * y[s])
        if g[s] > max_ and alpha[s] < Bs:
            max_ = g[s]
            selected = s
        inc(it)

    return selected


cdef int _argmin(np.ndarray[double, ndim=1] y,
                 np.ndarray[double, ndim=1, mode='c'] g,
                 list[long]& support_set,
                 np.ndarray[double, ndim=1, mode='c'] alpha,
                 double C):

    cdef int s, selected
    cdef double min_ = DBL_MAX
    cdef double As

    cdef list[long].iterator it = support_set.begin()
    while it != support_set.end():
        s = deref(it)
        As = min(0, C * y[s])
        if g[s] < min_ and alpha[s] > As:
            min_ = g[s]
            selected = s
        inc(it)

    return selected


cdef void _update(np.ndarray[double, ndim=2, mode='c'] X,
                  np.ndarray[double, ndim=1] y,
                  Kernel kernel,
                  np.ndarray[double, ndim=1, mode='c'] g,
                  list[long]& support_set,
                  np.ndarray[double, ndim=1, mode='c'] alpha,
                  int i,
                  int j,
                  double Aj,
                  double Bi):

    cdef double Kii, Kjj, Kij, Kis, Kjs

    # Need only three elements.
    Kii = kernel.compute_self(X, i)
    Kjj = kernel.compute_self(X, j)
    Kij = kernel.compute(X, i, X, j)
    cdef double lambda_ = min((g[i] - g[j]) / (Kii + Kjj - 2 * Kij),
                              min(Bi - alpha[i], alpha[j] - Aj))
    alpha[i] += lambda_
    alpha[j] -= lambda_


    cdef int s
    cdef list[long].iterator it = support_set.begin()
    while it != support_set.end():
        s = deref(it)
        # Need the ith and jth column (support vectors only)
        Kis = kernel.compute(X, s, X, i)
        Kjs = kernel.compute(X, s, X, j)
        g[s] -= lambda_ * (Kis - Kjs)
        inc(it)


cdef void _process(int k,
                   np.ndarray[double, ndim=2, mode='c'] X,
                   np.ndarray[double, ndim=1] y,
                   Kernel kernel,
                   list[long]& support_set,
                   np.ndarray[long, ndim=1, mode='c'] support_vectors,
                   np.ndarray[double, ndim=1, mode='c'] alpha,
                   np.ndarray[double, ndim=1, mode='c'] g,
                   double C,
                   double tau):

    if support_vectors[k]:
        return

    alpha[k] = 0

    cdef int s, j, i
    cdef double pred = 0

    cdef list[long].iterator it = support_set.begin()
    while it != support_set.end():
        s = deref(it)
        # Iterate over k-th column (support vectors only)
        pred += alpha[s] * kernel.compute(X, s, X, k)
        inc(it)

    g[k] = y[k] - pred

    support_set.push_back(k)
    support_vectors[k] = 1

    if y[k] == 1:
        i = k
        j = _argmin(y, g, support_set, alpha, C)
    else:
        j = k
        i = _argmax(y, g, support_set, alpha, C)

    cdef double Aj = min(0, C * y[j])
    cdef double Bi = max(0, C * y[i])

    cdef int violating_pair
    violating_pair = alpha[i] < Bi and alpha[j] > Aj and g[i] - g[j] > tau

    if not violating_pair:
        return

    _update(X, y, kernel, g, support_set, alpha, i, j, Aj, Bi)



cdef void _reprocess(np.ndarray[double, ndim=2, mode='c'] X,
                     np.ndarray[double, ndim=1] y,
                     Kernel kernel,
                     list[long]& support_set,
                     np.ndarray[long, ndim=1, mode='c'] support_vectors,
                     np.ndarray[double, ndim=1, mode='c'] alpha,
                     np.ndarray[double, ndim=1, mode='c'] g,
                     np.ndarray[double, ndim=1, mode='c'] b,
                     np.ndarray[double, ndim=1, mode='c'] delta,
                     double C,
                     double tau):
    cdef i = _argmax(y, g, support_set, alpha, C)
    cdef j = _argmin(y, g, support_set, alpha, C)

    cdef double Aj = min(0, C * y[j])
    cdef double Bi = max(0, C * y[i])

    cdef int violating_pair
    violating_pair = alpha[i] < Bi and alpha[j] > Aj and g[i] - g[j] > tau

    if not violating_pair:
        return

    _update(X, y, kernel, g, support_set, alpha, i, j, Aj, Bi)

    i = _argmax(y, g, support_set, alpha, C)
    j = _argmin(y, g, support_set, alpha, C)

    cdef int s, k = 0
    cdef int n_removed = 0

    cdef list[long].iterator it = support_set.begin()
    while it != support_set.end():
        s = deref(it)

        if alpha[s] == 0:
            if (y[s] == -1 and g[s] >= g[i]) or (y[s] == 1 and g[s] <= g[j]):
                it = support_set.erase(it)
                support_vectors[s] = 0
                n_removed += 1
                continue

        inc(it)

    b[0] = (g[i] + g[j]) / 2
    delta[0] = g[i] - g[j]


cdef void _boostrap(index,
                    np.ndarray[double, ndim=2, mode='c'] X,
                    np.ndarray[double, ndim=1] y,
                    Kernel kernel,
                    list[long]& support_set,
                    np.ndarray[long, ndim=1, mode='c'] support_vectors,
                    np.ndarray[double, ndim=1, mode='c'] alpha,
                    np.ndarray[double, ndim=1, mode='c'] g,
                    double* col_data,
                    rs):
    cdef np.ndarray[long, ndim=1, mode='c'] A = index
    cdef int n_pos = 0
    cdef int n_neg = 0
    cdef int i, s
    cdef Py_ssize_t n_samples = index.shape[0]

    rs.shuffle(A)

    for i in xrange(n_samples):
        s = A[i]

        if (y[s] == -1 and n_neg < 5) or (y[s] == 1 and n_pos < 5):
            support_set.push_back(s)
            support_vectors[s] = 1

            alpha[s] = y[s]
            # Entire sth column
            kernel.compute_column_ptr(X, X, s, col_data)

            for j in xrange(n_samples):
                # All elements of the s-th column.
                g[j] -= y[s] * col_data[j]

            if y[s] == -1:
                n_neg += 1
            else:
                n_pos += 1

        if (n_neg == 5 and n_pos == 5):
            break


cdef void _boostrap_warm_start(index,
                               np.ndarray[double, ndim=2, mode='c']X,
                               np.ndarray[double, ndim=1]y,
                               Kernel kernel,
                               list[long]& support_set,
                               np.ndarray[long, ndim=1, mode='c'] support_vectors,
                               np.ndarray[double, ndim=1, mode='c'] alpha,
                               np.ndarray[double, ndim=1, mode='c'] g,
                               double* col_data):
    cdef np.ndarray[long, ndim=1, mode='c'] A = index
    cdef int i, s
    cdef Py_ssize_t n_samples = index.shape[0]


    for i in xrange(n_samples):
        if alpha[i] != 0:
            kernel.compute_column_ptr(X, X, i, col_data)

            for j in xrange(n_samples):
                # All elements of the i-th column.
                g[j] -= alpha[i] * col_data[j]

            support_set.push_back(i)
            support_vectors[i] = 1


def _lasvm(np.ndarray[double, ndim=1, mode='c'] alpha,
           np.ndarray[double, ndim=2, mode='c'] X,
           np.ndarray[double, ndim=1] y,
           Kernel kernel,
           selection,
           int search_size,
           termination,
           int sv_upper_bound,
           double tau,
           int finish_step,
           double C,
           int max_iter,
           rs,
           int verbose,
           int warm_start):

    cdef Py_ssize_t n_samples = X.shape[0]
    cdef Py_ssize_t n_features = X.shape[1]

    cdef np.ndarray[long, ndim=1, mode='c'] A
    A = np.arange(n_samples)

    cdef list[long] support_set

    cdef np.ndarray[long, ndim=1, mode='c'] support_vectors
    support_vectors = np.zeros(n_samples, dtype=np.int64)

    cdef np.ndarray[double, ndim=1, mode='c'] g
    g = y.copy()

    cdef np.ndarray[double, ndim=1, mode='c'] b
    b = np.zeros(1, dtype=np.float64)

    cdef np.ndarray[double, ndim=1, mode='c'] delta
    delta = np.zeros(1, dtype=np.float64)

    cdef double* col_data
    cdef np.ndarray[double, ndim=1, mode='c'] col
    col = np.zeros(n_samples, dtype=np.float64)
    col_data = <double*>col.data

    cdef int check_n_sv = termination == "n_sv"

    cdef int it, i, j, s, k, start
    cdef int n_pos, n_neg
    cdef int stop = 0

    cdef int select_method = get_select_method(selection)

    if warm_start:
        _boostrap_warm_start(A, X, y, kernel,
                             support_set, support_vectors, alpha, g, col_data)
    else:
        _boostrap(A, X, y, kernel,
                  support_set, support_vectors, alpha, g, col_data, rs)

    for it in xrange(max_iter):

        start = 0
        rs.shuffle(A)

        for i in xrange(n_samples):
            # Select a support vector candidate.
            s = select_sv(A, start, search_size, n_samples, select_method,
                          alpha, b[0], X, y, kernel,
                          support_set, support_vectors)

            # Attempt to add it.
            _process(s, X, y, kernel,
                     support_set, support_vectors, alpha, g, C, tau)

            # Remove blatant non support vectors.
            _reprocess(X, y, kernel,
                       support_set, support_vectors, alpha, g, b, delta, C, tau)

            # Exit if necessary.
            if check_n_sv and support_set.size() >= sv_upper_bound:
                stop = 1
                break

            start = update_start(start, select_method, search_size,
                                 n_samples, A, rs)

        # end for

        if stop:
            break

    if finish_step:
        while delta[0] > tau:
            _reprocess(X, y, kernel,
                       support_set, support_vectors, alpha,
                       g, b, delta, C, tau)

    # Return intercept.
    return b[0]
