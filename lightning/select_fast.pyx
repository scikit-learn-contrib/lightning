# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
#
# Author: Mathieu Blondel
# License: BSD

from cython.operator cimport dereference as deref, preincrement as inc

cdef extern from "math.h":
   double fabs(double)

cdef extern from "float.h":
   double DBL_MAX


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
                   np.ndarray[double, ndim=1, mode='c'] col,
                   int check_duplicates):

    if select_method <= 1: # permute or random
        return A[start]

    cdef int i = start
    cdef int n_visited = 0
    cdef int s, k, j
    cdef double score
    cdef double min_score = DBL_MAX
    cdef int selected = 0
    cdef list[int].iterator it
    cdef list[int]* support_set = kcache.support_set
    cdef int* support_vectors = kcache.support_vector

    while n_visited < search_size and i < max_size:
        s = A[i]

        if check_duplicates and support_vectors[s] >= 0:
            i += 1
            continue

        k = 0
        score = 0

        # Compute prediction.
        it = support_set.begin()
        kcache.compute_column_sv(X, X, s, col)
        while it != support_set.end():
            j = deref(it)
            score += alpha[j] * col[j]
            inc(it)

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

cdef int select_sv_precomputed(np.ndarray[int, ndim=1, mode='c'] A,
                               int start,
                               int search_size,
                               int max_size,
                               int select_method,
                               np.ndarray[double, ndim=1, mode='c'] errors,
                               KernelCache kcache,
                               int check_duplicates):

    if select_method <= 1: # permute or random
        return A[start]

    cdef int i = start
    cdef int n_visited = 0
    cdef int s
    cdef double score
    cdef double min_score = DBL_MAX
    cdef int selected = 0
    cdef int* support_vectors = kcache.support_vector

    while n_visited < search_size and i < max_size:
        s = A[i]

        if check_duplicates and support_vectors[s] >= 0:
            i += 1
            continue

        if select_method == 2: # active
            score = fabs(errors[s] - 1)
        elif select_method == 3: # loss
            score = -errors[s]

        if score < min_score:
            min_score = score
            selected = s

        n_visited += 1
        i += 1

    return selected


cdef int update_start(int start,
                      int select_method,
                      int search_size,
                      int active_size,
                      np.ndarray[int, ndim=1, mode='c'] index,
                      rs):

    # Update position and reshuffle if needed.
    if select_method: # others than permute
        start += search_size

        if start + search_size > active_size - 1:
            rs.shuffle(index[:active_size])
            start = 0
    else:
        start += 1

    return start
