# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
#
# Author: Mathieu Blondel
# License: BSD

import numpy as np

cimport numpy as np
from lightning.kernel_fast cimport Kernel

cdef extern from "math.h":
   double fabs(double)

cdef extern from "float.h":
   double DBL_MAX


cdef int _argmax(np.ndarray[double, ndim=1]y,
                 np.ndarray[double, ndim=1, mode='c']g,
                 np.ndarray[long, ndim=1, mode='c']support_set,
                 np.ndarray[double, ndim=1, mode='c']alpha,
                 double C):

    cdef int s, selected, k = 0
    cdef double max_ = -DBL_MAX
    cdef double Bs

    while support_set[k] != -1:
        s = support_set[k]
        Bs = max(0, C * y[s])
        if g[s] > max_ and alpha[s] < Bs:
            max_ = g[s]
            selected = s
        k += 1

    return selected


cdef int _argmin(np.ndarray[double, ndim=1]y,
                 np.ndarray[double, ndim=1, mode='c']g,
                 np.ndarray[long, ndim=1, mode='c']support_set,
                 np.ndarray[double, ndim=1, mode='c']alpha,
                 double C):

    cdef int s, selected, k = 0
    cdef double min_ = DBL_MAX
    cdef double As

    while support_set[k] != -1:
        s = support_set[k]
        As = min(0, C * y[s])
        if g[s] < min_ and alpha[s] > As:
            min_ = g[s]
            selected = s
        k += 1

    return selected


cdef void _update(np.ndarray[double, ndim=2, mode='c']X,
                  np.ndarray[double, ndim=1]y,
                  Kernel kernel,
                  np.ndarray[double, ndim=1, mode='c']g,
                  np.ndarray[long, ndim=1, mode='c']support_set,
                  np.ndarray[double, ndim=1, mode='c']alpha,
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

    cdef int s, k = 0
    while support_set[k] != -1:
        s = support_set[k]
        # Need only two elements per column.
        Kis = kernel.compute(X, i, X, s)
        Kjs = kernel.compute(X, j, X, s)
        g[s] -= lambda_ * (Kis - Kjs)
        k += 1


cdef int _process(int k,
                  np.ndarray[double, ndim=2, mode='c']X,
                  np.ndarray[double, ndim=1]y,
                  Kernel kernel,
                  np.ndarray[long, ndim=1, mode='c']support_set,
                  np.ndarray[long, ndim=1, mode='c']support_vectors,
                  np.ndarray[double, ndim=1, mode='c']alpha,
                  np.ndarray[double, ndim=1, mode='c']g,
                  double C,
                  double tau):

    if support_vectors[k]:
        return 0

    alpha[k] = 0

    cdef int s, j, i = 0
    cdef double pred = 0

    # Iterate over k-th column (support vectors only)
    while support_set[i] != -1:
        s = support_set[i]
        pred += alpha[s] * kernel.compute(X, s, X, k)
        i += 1

    g[k] = y[k] - pred

    support_set[i] = k
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
        return 0

    _update(X, y, kernel, g, support_set, alpha, i, j, Aj, Bi)

    return 1

cpdef _remove(np.ndarray[long, ndim=1, mode='c']support_set,
              int k):
    k += 1
    while(support_set[k] != -1):
        support_set[k-1] = support_set[k]
        k += 1
    support_set[k-1] = -1

cdef int _reprocess(np.ndarray[double, ndim=2, mode='c']X,
                    np.ndarray[double, ndim=1]y,
                    Kernel kernel,
                    np.ndarray[long, ndim=1, mode='c']support_set,
                    np.ndarray[long, ndim=1, mode='c']support_vectors,
                    np.ndarray[double, ndim=1, mode='c']alpha,
                    np.ndarray[double, ndim=1, mode='c']g,
                    np.ndarray[double, ndim=1, mode='c']b,
                    np.ndarray[double, ndim=1, mode='c']delta,
                    double C,
                    double tau):
    cdef i = _argmax(y, g, support_set, alpha, C)
    cdef j = _argmin(y, g, support_set, alpha, C)

    cdef double Aj = min(0, C * y[j])
    cdef double Bi = max(0, C * y[i])

    cdef int violating_pair
    violating_pair = alpha[i] < Bi and alpha[j] > Aj and g[i] - g[j] > tau

    if not violating_pair:
        return 0

    _update(X, y, kernel, g, support_set, alpha, i, j, Aj, Bi)

    i = _argmax(y, g, support_set, alpha, C)
    j = _argmin(y, g, support_set, alpha, C)

    cdef int s, k = 0
    cdef int n_removed = 0

    while support_set[k] != -1:
        s = support_set[k]

        if alpha[s] == 0:
            if (y[s] == -1 and g[s] >= g[i]) or (y[s] == 1 and g[s] <= g[j]):
                _remove(support_set, k)
                support_vectors[s] = 0
                n_removed += 1
                k -= 1

        k += 1

    b[0] = (g[i] + g[j]) / 2
    delta[0] = g[i] - g[j]

    return n_removed

cdef int _boostrap(index,
                   np.ndarray[double, ndim=2, mode='c']X,
                   np.ndarray[double, ndim=1]y,
                   Kernel kernel,
                   np.ndarray[long, ndim=1, mode='c']support_set,
                   np.ndarray[long, ndim=1, mode='c']support_vectors,
                   np.ndarray[double, ndim=1, mode='c']alpha,
                   np.ndarray[double, ndim=1, mode='c']g,
                   double *col_data,
                   rs):
    cdef np.ndarray[long, ndim=1, mode='c'] A = index
    cdef int n_pos = 0
    cdef int n_neg = 0
    cdef int i, s, k = 0
    cdef Py_ssize_t n_samples = index.shape[0]

    rs.shuffle(A)

    for i in xrange(n_samples):
        s = A[i]

        if (y[s] == -1 and n_neg < 5) or (y[s] == 1 and n_pos < 5):
            support_set[k] = s
            support_vectors[s] = 1
            k += 1

            alpha[s] = y[s]
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

    return n_pos + n_neg


cdef int _boostrap_warm_start(index,
                              np.ndarray[double, ndim=2, mode='c']X,
                              np.ndarray[double, ndim=1]y,
                              Kernel kernel,
                              np.ndarray[long, ndim=1, mode='c']support_set,
                              np.ndarray[long, ndim=1, mode='c']support_vectors,
                              np.ndarray[double, ndim=1, mode='c']alpha,
                              np.ndarray[double, ndim=1, mode='c']g,
                              double* col_data):
    cdef np.ndarray[long, ndim=1, mode='c'] A = index
    cdef int i, s, k = 0
    cdef Py_ssize_t n_samples = index.shape[0]


    for i in xrange(n_samples):
        if alpha[i] != 0:
            kernel.compute_column_ptr(X, X, i, col_data)

            for j in xrange(n_samples):
                # All elements of the i-th column.
                g[j] -= alpha[i] * col_data[j]

            support_set[k] = i
            support_vectors[i] = 1
            k += 1


cdef int get_select_method(selection):
    if selection == "permute":
        return 0
    elif selection == "random":
        return 1
    elif selection == "active":
        return 2
    elif selection == "loss":
        return 3
    else:
        raise ValueError("Wrong selection method.")


cdef int select(np.ndarray[long, ndim=1, mode='c'] A,
                int start,
                int search_size,
                int max_size,
                int select_method,
                np.ndarray[double, ndim=1, mode='c']alpha,
                double b,
                np.ndarray[double, ndim=2, mode='c'] X,
                np.ndarray[double, ndim=1]y,
                Kernel kernel,
                np.ndarray[long, ndim=1, mode='c']support_set,
                np.ndarray[long, ndim=1, mode='c']support_vectors):

    if select_method <= 1: # permute or random
        return A[start]

    cdef int i = start
    cdef int n_visited = 0
    cdef int s, k, j
    cdef double score
    cdef double min_score = DBL_MAX
    cdef int selected = 0

    while n_visited < search_size and i < max_size:
        s = A[i]

        # Only non support vectors are candidates.
        if support_vectors[s]:
            i += 1
            continue

        k = 0
        score = 0

        # Compute prediction.
        while support_set[k] != -1:
            j = support_set[k]
            score += alpha[j] * kernel.compute(X, j, X, s)
            k += 1

        score += b

        if select_method == 2: # active
            score = fabs(score)
        elif select_method == 3: # loss
            score *= y[s]

        if score < min_score:
            min_score = score
            selected = s

        n_visited += 1
        i += 1

    return selected

def _lasvm(np.ndarray[double, ndim=1, mode='c']alpha,
           np.ndarray[double, ndim=2, mode='c'] X,
           np.ndarray[double, ndim=1]y,
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

    cdef np.ndarray[long, ndim=1, mode='c'] support_set
    support_set = np.zeros(n_samples + 1, dtype=np.int64)
    support_set -= 1

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
    cdef int n_sv
    cdef int stop = 0

    cdef int select_method = get_select_method(selection)

    if warm_start:
        n_sv = _boostrap_warm_start(A, X, y, kernel,
                                    support_set, support_vectors, alpha, g,
                                    col_data)
    else:
        n_sv = _boostrap(A, X, y, kernel,
                         support_set, support_vectors, alpha, g, col_data, rs)

    for it in xrange(max_iter):

        start = 0
        rs.shuffle(A)

        for i in xrange(n_samples):
            # Select a support vector candidate.
            s = select(A, start, search_size, n_samples, select_method,
                       alpha, b[0], X, y, kernel, support_set, support_vectors)

            # Attempt to add it.
            n_sv += _process(s, X, y, kernel,
                             support_set, support_vectors, alpha, g, C, tau)

            # Remove blatant non support vectors.
            n_sv -=_reprocess(X, y, kernel,
                              support_set, support_vectors, alpha,
                              g, b, delta, C, tau)

            # Exit if necessary.
            if n_sv == n_samples or (check_n_sv and n_sv > sv_upper_bound):
                stop = 1
                break

            # Update position and reshuffle if needed.
            if select_method: # others than permute
                start += search_size

                if start + search_size > n_samples - 1:
                    rs.shuffle(A)
                    start = 0
            else:
                start += 1

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
