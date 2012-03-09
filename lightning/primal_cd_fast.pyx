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

def _primal_cd_l2svm_l1r(np.ndarray[double, ndim=1, mode='c']w,
                         X,
                         np.ndarray[double, ndim=1] y,
                         Kernel kernel,
                         int linear_kernel,
                         double C,
                         int max_iter,
                         rs,
                         double tol,
                         int verbose):

    cdef Py_ssize_t n_samples = X.shape[0]
    cdef Py_ssize_t n_features = X.shape[1]

    cdef np.ndarray[double, ndim=2, mode='fortran'] Xf
    cdef np.ndarray[double, ndim=2, mode='c'] Xc
    cdef np.ndarray[double, ndim=1, mode='c'] b

    if linear_kernel:
        Xf = X
        b = 1 - y * np.dot(X, w)
    else:
        Xc = X
        b = np.ones(n_samples, dtype=np.float64)
        n_features = n_samples

    cdef int j, s, it, i = 0
    cdef int active_size = n_features
    cdef int max_num_linesearch = 20

    cdef double sigma = 0.01
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

    cdef np.ndarray[long, ndim=1, mode='c'] index
    index = np.arange(n_features)

    cdef double* col_data
    cdef double* col_ro
    cdef np.ndarray[double, ndim=1, mode='c'] col
    col = np.zeros(n_samples, dtype=np.float64)
    col_data = <double*>col.data


    for it in xrange(max_iter):
        Lpmax_new = 0
        rs.shuffle(index[:active_size])

        s = 0
        while s < active_size:
            j = index[s]
            Lp = 0
            Lpp = 0
            xj_sq = 0

            if linear_kernel:
                col_ro = (<double*>Xf.data) + j * n_samples
            else:
                kernel.compute_column_ptr(Xc, Xc, j, col_data)
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
                s += 1
                continue

            wj_abs = fabs(w[j])
            delta = fabs(w[j] + d) - wj_abs + Lp * d
            z_old = 0
            z = d

            for num_linesearch in xrange(max_num_linesearch):
                z_diff = z_old - z
                cond = fabs(w[j] + z) - wj_abs - sigma * delta

                appxcond = xj_sq * z * z + Lp * z + cond

                # Avoid line search if possible.
                if appxcond <= 0:
                    for i in xrange(n_samples):
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
                    z *= 0.5
                    delta *= 0.5

            # end for num_linesearch

            w[j] += z

            # recompute b[] if line search takes too many steps
            #if num_linesearch >= max_num_linesearch:
                #b[:] = 1

                #for i in xrange(n_features):
                    #if w[i] == 0:
                        #continue

                    #for ind in xrange(n_samples):
                        #b[ind] -= w[i] * X[ind, j]
                ## end for


            s += 1
        # while active_size

        if it == 0:
            Lpmax_init = Lpmax_new

        if Lpmax_new <= tol * Lpmax_init:
            if active_size == n_features:
                if verbose:
                    print "Converged at iteration", it
                break
            else:
                active_size = n_features
                Lpmax_old = DBL_MAX
                continue

        Lpmax_old = Lpmax_new

    # end for while max_iter

    return w


#cdef double _recompute(X,
                       #np.ndarray[double, ndim=1, mode='c'] w,
                       #double C,
                       #np.ndarray[double, ndim=1, mode='c'] b,
                       #int ind):
    #cdef Py_ssize_t n_samples = X.shape[0]
    #cdef Py_ssize_t n_features = X.shape[1]

    #cdef np.ndarray[double, ndim=2, mode='fortran'] X
    #X = X

    #b[:] = 1

    #cdef int i, j

    #for j in xrange(n_features):
        #for i in xrange(n_samples):
            #b[i] -= w[j] * X[i, j]

    #cdef double loss = 0

    ## Can iterate over the non-zero samples only
    #for i in xrange(n_samples):
        #if b[i] > 0:
            #loss += b[i] * b[i] * C # Cp

    #return loss


def _primal_cd_l2svm_l2r(np.ndarray[double, ndim=1] w,
                         X,
                         np.ndarray[double, ndim=1] y,
                         Kernel kernel,
                         int linear_kernel,
                         double C,
                         int max_iter,
                         rs,
                         double tol,
                         int verbose):

    cdef Py_ssize_t n_samples = X.shape[0]
    cdef Py_ssize_t n_features = X.shape[1]

    cdef np.ndarray[double, ndim=2, mode='fortran'] Xf
    cdef np.ndarray[double, ndim=2, mode='c'] Xc
    cdef np.ndarray[double, ndim=1, mode='c'] b

    if linear_kernel:
        Xf = X
        b = 1 - y * np.dot(X, w)
    else:
        Xc = X
        b = np.ones(n_samples, dtype=np.float64)
        n_features = n_samples

    cdef int i, j, s, step, it
    cdef double d, old_d, Dp, Dpmax, Dpp, loss, new_loss
    cdef double sigma = 0.01
    cdef double xj_sq, val, ddiff, tmp, bound

    cdef np.ndarray[long, ndim=1, mode='c'] index
    index = np.arange(n_features)

    cdef double* col_data
    cdef double* col_ro
    cdef np.ndarray[double, ndim=1, mode='c'] col
    col = np.zeros(n_samples, dtype=np.float64)
    col_data = <double*>col.data

    for it in xrange(max_iter):
        Dpmax = 0

        rs.shuffle(index)

        for s in xrange(n_features):
            j = index[s]
            Dp = 0
            Dpp = 0
            loss = 0

            if linear_kernel:
                col_ro = (<double*>Xf.data) + j * n_samples
            else:
                kernel.compute_column_ptr(Xc, Xc, j, col_data)
                col_ro = col_data

            # Iterate over samples that have the feature
            xj_sq = 0
            for i in xrange(n_samples):
                val = col_ro[i] * y[i]
                col[i] = val
                xj_sq += val * val

                if b[i] > 0:
                    Dp -= b[i] * val * C
                    Dpp += val * val * C
                    if val != 0:
                        loss += b[i] * b[i] * C

            bound = (2 * C * xj_sq + 1) / 2.0 + sigma

            Dp = w[j] + 2 * Dp
            Dpp = 1 + 2 * Dpp

            if fabs(Dp) > Dpmax:
                Dpmax = fabs(Dp)

            if fabs(Dp/Dpp) <= 1e-12:
                continue

            d = -Dp / Dpp
            old_d = 0
            step = 0

            while step < 100:
                ddiff = old_d - d
                step += 1

                if Dp/d + bound <= 0:
                    for i in xrange(n_samples):
                        b[i] += ddiff * col[i]
                    break

                # Recompute if line search too many times
                #if step % 10 == 0:
                    #loss = _recompute(X, w, C, b, j)
                    #for i in xrange(n_samples):
                        #b[i] -= old_d * col[i]

                new_loss = 0

                for i in xrange(n_samples):
                    tmp = b[i] + ddiff * col[i]
                    b[i] = tmp
                    if tmp > 0:
                        new_loss += tmp * tmp * C

                old_d = d

                if w[j] * d + (0.5 + sigma) * d * d + new_loss - loss <= 0:
                    break
                else:
                    d /= 2

            # end while (line search)

            w[j] += d

        # end for (iterate over features)

        if Dpmax < tol:
            if verbose >= 1:
                print "Converged at iteration", it
            break

    return w

