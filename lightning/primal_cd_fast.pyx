# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
#
# Author: Mathieu Blondel
# License: BSD

import sys

from cython.operator cimport dereference as deref
from cython.operator cimport preincrement as inc
from cython.operator cimport predecrement as dec

from libcpp.list cimport list
from libcpp.vector cimport vector

import numpy as np
cimport numpy as np

from lightning.kernel_fast cimport KernelCache, Kernel
from lightning.select_fast cimport get_select_method, select_sv, update_start

cdef extern from "math.h":
   double fabs(double)

cdef extern from "float.h":
   double DBL_MAX


def _primal_cd_l2svm_l1r(self,
                         np.ndarray[double, ndim=1, mode='c'] w,
                         np.ndarray[double, ndim=1, mode='c'] b,
                         X,
                         np.ndarray[double, ndim=1] y,
                         KernelCache kcache,
                         int linear_kernel,
                         selection,
                         int search_size,
                         termination,
                         int sv_upper_bound,
                         double C,
                         int max_iter,
                         rs,
                         double tol,
                         callback,
                         int verbose):

    cdef Py_ssize_t n_samples = X.shape[0]
    cdef Py_ssize_t n_features = X.shape[1]

    cdef np.ndarray[double, ndim=2, mode='fortran'] Xf
    cdef np.ndarray[double, ndim=2, mode='c'] Xc

    if linear_kernel:
        Xf = X
    else:
        Xc = X
        n_features = n_samples

    cdef int j, s, t, start, i = 0
    cdef int active_size = n_features
    cdef int max_num_linesearch = 20

    cdef double sigma = 0.01
    cdef double beta = 0.5
    cdef double d, Lp, Lpp, Lpp_wj
    cdef double Lpmax_old = DBL_MAX
    cdef double Lpmax_new
    cdef double Lpmax_init
    cdef double z, z_old, z_diff
    cdef double Lj_zero, Lj_z
    cdef double appxcond, cond
    cdef double val, val_sq
    cdef double Lp_p, Lp_n, violation
    cdef double delta, b_new, b_add
    cdef double xj_sq
    cdef double wj_abs

    cdef np.ndarray[int, ndim=1, mode='c'] index
    index = np.arange(n_features, dtype=np.int32)

    cdef double* col_data
    cdef double* col_ro
    cdef np.ndarray[double, ndim=1, mode='c'] col
    col = np.zeros(n_samples, dtype=np.float64)
    col_data = <double*>col.data

    cdef list[int].iterator it

    cdef int check_n_sv = termination in ("n_sv", "n_nz_coef")
    cdef int check_convergence = termination == "convergence"
    cdef int stop = 0
    cdef int select_method = get_select_method(selection)
    cdef int permute = selection == "permute" or linear_kernel
    cdef int has_callback = callback is not None

    # FIXME: would be better to store the support indices in the class.
    if not linear_kernel:
        for j in xrange(n_features):
            if w[j] != 0:
                kcache.add_sv(j)

    for t in xrange(max_iter):
        if verbose >= 1:
            print "\nIteration", t

        Lpmax_new = 0
        rs.shuffle(index[:active_size])

        s = 0
        start = 0

        while s < active_size:
            if permute:
                j = index[s]
            else:
                j = select_sv(index, start, search_size, active_size,
                              select_method, w, 0, Xc, y, kcache, col)

            Lp = 0
            Lpp = 0
            xj_sq = 0

            if linear_kernel:
                col_ro = (<double*>Xf.data) + j * n_samples
            else:
                kcache.compute_column(Xc, Xc, j, col)
                col_ro = col_data

            for i in xrange(n_samples):
                val = col_ro[i] * y[i]
                col[i] = val
                val_sq = val * val
                if b[i] > 0:
                    Lp -= val * b[i]
                    Lpp += val_sq
                xj_sq += val_sq
            # end for

            xj_sq *= C
            Lp *= 2 * C

            Lpp *= 2 * C
            Lpp = max(Lpp, 1e-12)

            Lp_p = Lp + 1
            Lp_n = Lp - 1
            violation = 0

            # Shrinking.
            if w[j] == 0:
                if Lp_p < 0:
                    violation = -Lp_p
                elif Lp_n > 0:
                    violation = Lp_n
                elif Lp_p > Lpmax_old / n_samples and Lp_n < -Lpmax_old / n_samples:
                    active_size -= 1
                    index[s], index[active_size] = index[active_size], index[s]
                    start = update_start(start, select_method, search_size,
                                         active_size, index, rs)
                    # Jump w/o incrementing s so as to use the swapped sample.
                    continue
            elif w[j] > 0:
                violation = fabs(Lp_p)
            else:
                violation = fabs(Lp_n)

            Lpmax_new = max(Lpmax_new, violation)

            # Obtain Newton direction d.
            Lpp_wj = Lpp * w[j]
            if Lp_p <= Lpp_wj:
                d = -Lp_p / Lpp
            elif Lp_n >= Lpp_wj:
                d = -Lp_n / Lpp
            else:
                d = -w[j]

            if fabs(d) < 1.0e-12:
                start = update_start(start, select_method, search_size,
                                     active_size, index, rs)
                s += 1
                continue

            wj_abs = fabs(w[j])
            delta = fabs(w[j] + d) - wj_abs + Lp * d
            z_old = 0
            z = d

            # Check z = lambda*d for lambda = 1, beta, beta^2 such that
            # sufficient decrease condition is met.
            for num_linesearch in xrange(max_num_linesearch):
                # Reversed because of the minus in b[i] = 1 - y_i w^T x_i.
                z_diff = z_old - z
                cond = fabs(w[j] + z) - wj_abs - sigma * delta

                appxcond = xj_sq * z * z + Lp * z + cond

                # Avoid line search if possible.
                if appxcond <= 0:
                    for i in xrange(n_samples):
                        # Need to remove the old z and had the new one.
                        b[i] += z_diff * col[i]
                    break

                if num_linesearch == 0:
                    Lj_zero = 0
                    Lj_z = 0

                    # L_j = \sum_{b_i > 0} b_i ^2
                    for i in xrange(n_samples):
                        if b[i] > 0:
                            Lj_zero += b[i] * b[i]

                        b_new = b[i] + z_diff * col[i]
                        b[i] = b_new

                        if b_new > 0:
                            Lj_z += b_new * b_new

                    Lj_zero *= C
                    Lj_z *= C
                else:
                    Lj_z = 0

                    for i in xrange(n_samples):
                        b_new = b[i] + z_diff * col[i]
                        b[i] = b_new
                        if b_new > 0:
                            Lj_z += b_new * b_new

                    Lj_z *= C

                cond = cond + Lj_z - Lj_zero
                if cond <= 0:
                    break
                else:
                    z_old = z
                    z *= beta
                    delta *= beta

            # end for num_linesearch

            w[j] += z

            # Update support set.
            if not linear_kernel:
                if w[j] != 0:
                    kcache.add_sv(j)
                elif w[j] == 0:
                    kcache.remove_sv(j)

            # Exit if necessary.
            if check_n_sv and kcache.n_sv() >= sv_upper_bound:
                stop = 1
                break

            start = update_start(start, select_method, search_size,
                                 active_size, index, rs)

            # Callback
            if has_callback and s % 100 == 0:
                ret = callback(self)
                if ret is not None:
                    stop = 1
                    break

            if verbose >= 1 and s % 100 == 0:
                sys.stdout.write(".")
                sys.stdout.flush()

            s += 1
        # while active_size

        if stop:
            break

        if t == 0:
            Lpmax_init = Lpmax_new

        if check_convergence and Lpmax_new <= tol * Lpmax_init:
            if active_size == n_features:
                if verbose >= 1:
                    print "\nConverged at iteration", t
                break
            else:
                active_size = n_features
                Lpmax_old = DBL_MAX
                continue

        Lpmax_old = Lpmax_new

    # end for while max_iter

    if verbose >= 1:
        print

    return w


def _primal_cd_l2svm_l2r(self,
                         np.ndarray[double, ndim=1, mode='c'] w,
                         np.ndarray[double, ndim=1, mode='c'] b,
                         X,
                         A,
                         np.ndarray[double, ndim=1] y,
                         KernelCache kcache,
                         int linear_kernel,
                         termination,
                         int sv_upper_bound,
                         double C,
                         int max_iter,
                         rs,
                         double tol,
                         callback,
                         int verbose):

    cdef Py_ssize_t n_samples = X.shape[0]
    cdef Py_ssize_t n_features = X.shape[1]

    cdef np.ndarray[double, ndim=2, mode='fortran'] Xf
    cdef np.ndarray[double, ndim=2, mode='c'] Xc
    cdef np.ndarray[double, ndim=2, mode='c'] Ac

    if linear_kernel:
        Xf = X
    else:
        Xc = X
        Ac = A
        n_features = A.shape[0]

    cdef int i, j, s, step, t
    cdef double z, z_old, z_diff,
    cdef double Dp, Dpmax, Dpp, Dj_zero, Dj_z
    cdef double sigma = 0.01
    cdef double xj_sq, val, b_new, bound

    cdef np.ndarray[int, ndim=1, mode='c'] index
    index = np.arange(n_features, dtype=np.int32)

    cdef double* col_data
    cdef double* col_ro
    cdef np.ndarray[double, ndim=1, mode='c'] col
    col = np.zeros(n_samples, dtype=np.float64)
    col_data = <double*>col.data

    cdef int check_n_sv = termination in ("n_sv", "n_nz_coef")
    cdef int check_convergence = termination == "convergence"
    cdef int has_callback = callback is not None
    cdef int stop = 0
    cdef int n_sv = 0


    for t in xrange(max_iter):
        if verbose >= 1:
            print "\nIteration", t

        Dpmax = 0

        rs.shuffle(index)

        for s in xrange(n_features):
            j = index[s]
            Dp = 0
            Dpp = 0
            Dj_zero = 0

            if linear_kernel:
                col_ro = (<double*>Xf.data) + j * n_samples
            else:
                kcache.compute_column(Xc, Ac, j, col)
                col_ro = col_data

            # Iterate over samples that have the feature
            xj_sq = 0
            for i in xrange(n_samples):
                val = col_ro[i] * y[i]
                col[i] = val
                xj_sq += val * val

                if b[i] > 0:
                    Dp -= b[i] * val
                    Dpp += val * val
                    Dj_zero += b[i] * b[i]

            bound = (2 * C * xj_sq + 1) / 2.0 + sigma

            Dp = w[j] + 2 * C * Dp
            Dpp = 1 + 2 * C * Dpp
            Dj_zero *= C

            if fabs(Dp) > Dpmax:
                Dpmax = fabs(Dp)

            if fabs(Dp/Dpp) <= 1e-12:
                continue

            d = -Dp / Dpp
            z_old = 0
            z = d

            for step in xrange(100):
                z_diff = z_old - z

                if Dp/z + bound <= 0:
                    for i in xrange(n_samples):
                        b[i] += z_diff * col[i]
                    break

                Dj_z = 0

                for i in xrange(n_samples):
                    b_new = b[i] + z_diff * col[i]
                    b[i] = b_new
                    if b_new > 0:
                        Dj_z += b_new * b_new

                Dj_z *= C

                z_old = z

                if w[j] * z + (0.5 + sigma) * z * z + Dj_z - Dj_zero <= 0:
                    break
                else:
                    z /= 2

            # end while (line search)

            w[j] += z

            if w[j] != 0:
                n_sv += 1

            if check_n_sv and n_sv == sv_upper_bound:
                stop = 1
                break

            # Callback
            if has_callback and s % 100 == 0:
                ret = callback(self)
                if ret is not None:
                    stop = 1
                    break

            if verbose >= 1 and s % 100 == 0:
                sys.stdout.write(".")
                sys.stdout.flush()

        # end for (iterate over features)

        if stop:
            break

        if check_convergence and Dpmax < tol:
            if verbose >= 1:
                print "\nConverged at iteration", t
            break

    # for iterations

    if verbose >= 1:
        print

    return w


cpdef _C_lower_bound_kernel(np.ndarray[double, ndim=2, mode='c'] X,
                            np.ndarray[double, ndim=2, mode='c'] Y,
                            Kernel kernel,
                            search_size=None,
                            random_state=None):

    cdef int n_samples = X.shape[0]
    cdef int n = n_samples

    cdef int i, j, k, l
    cdef int n_vectors = Y.shape[1]

    cdef double val, max_ = -DBL_MAX

    cdef np.ndarray[int, ndim=1, mode='c'] ind
    ind = np.arange(n, dtype=np.int32)

    cdef np.ndarray[double, ndim=1, mode='c'] col
    col = np.zeros(n, dtype=np.float64)

    if search_size is not None:
        n = search_size
        random_state.shuffle(ind)

    for j in xrange(n):
        k = ind[j]

        for i in xrange(n_samples):
            col[i] = kernel.compute(X, i, X, k)

        for l in xrange(n_vectors):
            val = 0
            for i in xrange(n_samples):
                val += Y[i, l] * col[i]
            max_ = max(max_, fabs(val))

    return max_
