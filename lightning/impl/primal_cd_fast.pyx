# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: language_level=3
#
# Author: Mathieu Blondel
# License: BSD

import numpy as np
cimport numpy as np

from libc.math cimport fabs, exp, log, sqrt

from lightning.impl.randomkit.random_fast cimport RandomState
from lightning.impl.dataset_fast cimport ColumnDataset

DEF LOWER = 1e-2
DEF UPPER = 1e9


cdef inline void _swap(int* arr, int a, int b):
    cdef int tmp
    tmp = arr[a]
    arr[a] = arr[b]
    arr[b] = tmp


cdef class LossFunction:

    cdef int max_steps
    cdef double sigma
    cdef double beta
    cdef int verbose

    # L2 regularization

    cdef void solve_l2(self,
                       int j,
                       double C,
                       double alpha,
                       double *w,
                       ColumnDataset X,
                       double *y,
                       double *b,
                       double *violation):

        cdef double Dp, Dpp, Dj_zero, z, d
        cdef int i, ii, step, recompute
        cdef double z_diff, z_old, Dj_z, cond

        # Data pointers
        cdef double* data
        cdef int* indices
        cdef int n_nz

        # Retrieve column.
        X.get_column_ptr(j, &indices, &data, &n_nz)

        # Compute derivatives.
        self.derivatives(j, C, indices, data, n_nz, y, b,
                         &Dp, &Dpp, &Dj_zero)

        # Add regularization term.
        Dp = alpha * w[j] + Dp # first derivative
        Dpp = alpha + Dpp # second derivative

        if fabs(Dp/Dpp) <= 1e-12:
            return

        # Newton step
        d = -Dp / Dpp

        # Perform line search.
        z_old = 0
        z = d

        step = 1
        recompute = 0
        while True:
            z_diff = z_old - z

            # Update predictions / errors / residuals (depends on loss).
            self.update(j, z_diff, C, indices, data, n_nz,
                        y, b, &Dj_z)

            if step >= self.max_steps:
                if self.max_steps > 1:
                    if self.verbose >= 2:
                        print("Max steps reached during line search...")
                    recompute = 1
                break

            # Check decrease condition
            #   0.5 * alpha * (w + z e_j)^T (w + z e_j)
            # = 0.5 * alpha * w^T w + alpha * w_j z + 0.5 * alpha * z^2
            cond = alpha * w[j] * z + (0.5 * alpha + self.sigma) * z * z
            if cond + Dj_z - Dj_zero <= 0:
                break

            z_old = z
            z *= self.beta
            step += 1

        # Update weight w[j].
        w[j] += z

        # Recompute predictions / errors / residuals if needed.
        if recompute:
            self.recompute(X, y, w, b)

        # For L2-regularized problems, we can simply use the absolute value of
        # the gradient as violation.
        violation[0] = fabs(Dp)

    cdef void derivatives(self,
                          int j,
                          double C,
                          int *indices,
                          double *data,
                          int n_nz,
                          double *y,
                          double *b,
                          double *Lp,
                          double *Lpp,
                          double *L):
        raise NotImplementedError()

    cdef void update(self,
                     int j,
                     double z_diff,
                     double C,
                     int *indices,
                     double *data,
                     int n_nz,
                     double *y,
                     double *b,
                     double *L_new):
        raise NotImplementedError()

    cdef void recompute(self,
                        ColumnDataset X,
                        double* y,
                        double* w,
                        double* b):
        pass

    cdef void _lipschitz_constant(self,
                                  ColumnDataset X,
                                  double scale,
                                  double* out):

        cdef int n_samples = X.get_n_samples()
        cdef int n_features = X.get_n_features()
        cdef int i, j, ii

        # Data pointers
        cdef double* data
        cdef int* indices
        cdef int n_nz

        for j in xrange(n_features):
            X.get_column_ptr(j, &indices, &data, &n_nz)

            for ii in xrange(n_nz):
                i = indices[ii]

                out[j] += scale * data[ii] * data[ii]

    # L1 regularization

    cdef int solve_l1(self,
                      int j,
                      double C,
                      double alpha,
                      double *w,
                      int n_samples,
                      ColumnDataset X,
                      double *y,
                      double *b,
                      double Lcst,
                      double violation_old,
                      double *violation,
                      int shrinking):
        cdef double Lj_zero = 0
        cdef double Lp = 0
        cdef double Lpp = 0
        cdef double Lpp_wj, d, wj_abs
        cdef double cond
        cdef double Lj_z
        cdef int step, recompute

        # Data pointers
        cdef double* data
        cdef int* indices
        cdef int n_nz

        # Retrieve column.
        X.get_column_ptr(j, &indices, &data, &n_nz)

        # Compute derivatives
        self.derivatives(j, C, indices, data, n_nz, y, b,
                         &Lp, &Lpp, &Lj_zero)

        # User chose to run the algorithm without line search.
        if self.max_steps == 0:
            Lpp_max = Lcst

        Lpp = max(Lpp, 1e-12)

        # Add regularization term.
        Lp_p = Lp + alpha
        Lp_n = Lp - alpha
        violation[0] = 0

        # Violation and shrinking.
        if w[j] == 0:
            if Lp_p < 0:
                violation[0] = -Lp_p
            elif Lp_n > 0:
                violation[0] = Lp_n
            elif shrinking and \
                 Lp_p > violation_old / n_samples and \
                 Lp_n < -violation_old / n_samples:
                # Shrink!
                if self.verbose >= 3:
                    print("Shrink variable", j)
                return 1
        elif w[j] > 0:
            violation[0] = fabs(Lp_p)
        else:
            violation[0] = fabs(Lp_n)

        # Obtain Newton direction d.
        Lpp_wj = Lpp * w[j]
        if Lp_p <= Lpp_wj:
            d = -Lp_p / Lpp
        elif Lp_n >= Lpp_wj:
            d = -Lp_n / Lpp
        else:
            d = -w[j]

        if fabs(d) < 1.0e-12:
            return 0

        # Perform line search.
        # Check z = lambda*d for lambda = 1, beta, beta^2 until a
        # sufficient decrease condition is met.
        wj_abs = fabs(w[j])
        delta = alpha * (fabs(w[j] + d) - wj_abs) + Lp * d
        z_old = 0
        z = d

        step = 1
        recompute = 0
        while True:
            # Reversed because of the minus in b[i] = 1 - y_i w^T x_i.
            z_diff = z_old - z

            # Compute objective function value.
            self.update(j, z_diff, C, indices, data, n_nz, y, b, &Lj_z)

            if step >= self.max_steps:
                if self.max_steps > 1:
                    if self.verbose >= 2:
                        print("Max steps reached during line search...")
                    recompute = 1
                break

            # Check stopping condition.
            cond = alpha * (fabs(w[j] + z) - wj_abs) - self.sigma * delta
            if cond + Lj_z - Lj_zero <= 0:
                break

            z_old = z
            z *= self.beta
            delta *= self.beta
            step += 1

        # end for num_linesearch

        # Update weight w[j].
        w[j] += z

        # Recompute predictions / errors / residuals if needed.
        if recompute:
            self.recompute(X, y, w, b)

        return 0

    # L1/L2 regularization

    cdef int solve_l1l2(self,
                        int j,
                        double C,
                        double alpha,
                        np.ndarray[double, ndim=2, mode='c'] w,
                        int n_vectors,
                        ColumnDataset X,
                        int* y,
                        np.ndarray[double, ndim=2, mode='fortran'] Y,
                        int multiclass,
                        np.ndarray[double, ndim=2, mode='c'] b,
                        double Lcst,
                        double *g,
                        double *d,
                        double *d_old,
                        double* Z,
                        double violation_old,
                        double *violation,
                        int shrinking):

        cdef int n_samples = Y.shape[0]
        cdef int n_features = w.shape[1]
        cdef int i, k, ii, step, recompute
        cdef double scaling, delta, L, R_j, Lpp_max, dmax
        cdef double tmp, L_new, R_j_new
        cdef double L_tmp, Lpp_tmp
        cdef double* y_ptr = <double*>Y.data
        cdef double* b_ptr = <double*>b.data
        cdef double* w_ptr = <double*>w.data
        cdef double z_diff, g_norm
        cdef int nv = n_samples * n_vectors

        # getting DBL_MAX from float.h fails for some reason on
        # MSVC 9.0 (which is needed to compile on Python 2.7)
        # To avoid it we take this constant from numpy
        cdef double DBL_MAX = np.finfo(np.double).max

        # Data pointers
        cdef double* data
        cdef int* indices
        cdef int n_nz

        # Retrieve column.
        X.get_column_ptr(j, &indices, &data, &n_nz)

        # Compute partial gradient.
        if multiclass:
            self.derivatives_mc(j, C, n_samples, n_vectors, indices, data, n_nz,
                                y, b_ptr, g, Z, &L, &Lpp_max)
        else: # multi-task
            L = 0
            Lpp_max = -DBL_MAX

            for k in xrange(n_vectors):
                self.derivatives(j, C, indices, data, n_nz, y_ptr,
                                 b_ptr, &g[k], &Lpp_tmp, &L_tmp)
                L += L_tmp
                Lpp_max = max(Lpp_max, Lpp_tmp)
                y_ptr += n_samples
                b_ptr += n_samples

            # Make sure Lpp is not too small (negative) or too large.
            Lpp_max = min(max(Lpp_max, LOWER), UPPER)

        # User chose to run the algorithm without line search.
        if self.max_steps == 0:
            if Lcst == 0:
                return 0 # The corresponding column is entirely 0.
            Lpp_max = Lcst

        # Compute partial gradient norm and regularization term.
        g_norm = 0
        R_j = 0

        for k in xrange(n_vectors):
            g_norm += g[k] * g[k]
            R_j += w[k, j] * w[k, j]

        g_norm = sqrt(g_norm)
        R_j = sqrt(R_j)

        # Violation and shrinking.
        if R_j == 0:
            g_norm -= alpha
            if g_norm > 0:
                violation[0] = g_norm
            elif shrinking and \
                 g_norm + violation_old / nv <= 0:
                # Shrink!
                if self.verbose >= 2:
                    print("Shrink variable", j)
                return 1
        else:
            violation[0] = fabs(g_norm - alpha)

        # Compute vector to be projected and scaling factor.
        scaling = 0
        for k in xrange(n_vectors):
            d_old[k] = 0
            d[k] = w[k, j] - g[k] / Lpp_max
            scaling += d[k] * d[k]

        scaling = 1 - alpha / (Lpp_max * sqrt(scaling))

        if scaling < 0:
            scaling = 0

        # Project (proximity operator).
        delta = 0
        dmax = -DBL_MAX
        for k in xrange(n_vectors):
            # Difference between new and old solution.
            d[k] = scaling * d[k] - w[k, j]
            delta += d[k] * g[k]
            dmax = max(dmax, fabs(d[k]))

        # Do not bother update if update is too small.
        if dmax < 1e-12:
            return 0

        # Perform line search.
        step = 1
        recompute = 0
        while True:

            # Update predictions, normalizations and objective value.
            if multiclass:
                self.update_mc(C, n_samples, n_vectors, indices, data, n_nz,
                               y, b_ptr, d, d_old, Z, &L_new)
            else: # multi-task
                L_new = 0
                y_ptr = <double*>Y.data
                b_ptr = <double*>b.data

                for k in xrange(n_vectors):
                    z_diff = d_old[k] - d[k]
                    self.update(j, z_diff, C, indices, data, n_nz,
                                y_ptr, b_ptr, &L_tmp)
                    L_new += L_tmp
                    y_ptr += n_samples
                    b_ptr += n_samples

            if step >= self.max_steps:
                if self.max_steps > 1:
                    if self.verbose >= 2:
                        print("Max steps reached during line search...")
                    recompute = 1
                break

            # Compute regularization term.
            R_j_new = 0
            for k in xrange(n_vectors):
                tmp = w[k, j] + d[k]
                R_j_new += tmp * tmp
            R_j_new = sqrt(R_j_new)
            # R_new = R - R_j + R_j_new

            if step == 1:
                delta += alpha * (R_j_new - R_j)
                delta *= self.sigma

            # Check decrease condition
            if L_new - L + alpha * (R_j_new - R_j) <= delta:
                break

            delta *= self.beta
            for k in xrange(n_vectors):
                d_old[k] = d[k]
                d[k] *= self.beta
            step += 1

        # Update solution
        for k in xrange(n_vectors):
            w[k, j] += d[k]

        # Recompute errors if necessary.
        if recompute:
            if multiclass:
                self.recompute_mc(n_vectors, X, y, w, b)
            else: # multi-task
                y_ptr = <double*>Y.data
                b_ptr = <double*>b.data
                w_ptr = <double*>w.data

                for k in xrange(n_vectors):
                    self.recompute(X, y_ptr, w_ptr, b_ptr)
                    y_ptr += n_samples
                    b_ptr += n_samples
                    w_ptr += n_features

        return 0

    cdef void derivatives_mc(self,
                             int j,
                             double C,
                             int n_samples,
                             int n_vectors,
                             int* indices,
                             double *data,
                             int n_nz,
                             int* y,
                             double* b,
                             double* g,
                             double* Z,
                             double* L,
                             double* Lpp_max):
        raise NotImplementedError()

    cdef void update_mc(self,
                        double C,
                        int n_samples,
                        int n_vectors,
                        int* indices,
                        double *data,
                        int n_nz,
                        int* y,
                        double *b,
                        double *d,
                        double *d_old,
                        double* Z,
                        double* L_new):
        raise NotImplementedError()

    cdef void recompute_mc(self,
                           int n_vectors,
                           ColumnDataset X,
                           int* y,
                           np.ndarray[double, ndim=2, mode='c'] w,
                           np.ndarray[double, ndim=2, mode='c'] b):
        pass

    cdef void lipschitz_constant_mt(self,
                                    int n_vectors,
                                    ColumnDataset X,
                                    double C,
                                    double* out):
        raise NotImplementedError()

    cdef void lipschitz_constant_mc(self,
                                    int n_vectors,
                                    ColumnDataset X,
                                    double C,
                                    double* out):
        raise NotImplementedError()


cdef class Squared(LossFunction):

    def __init__(self, verbose=0):
        # Squared loss enjoys closed form solution.
        # Therefore a single step is enough (no line search needed).
        self.max_steps = 1
        self.beta = 0.5
        self.sigma = 0.01
        self.verbose = verbose

    cdef void derivatives(self,
                          int j,
                          double C,
                          int *indices,
                          double *data,
                          int n_nz,
                          double *y,
                          double *b,
                          double *Lp,
                          double *Lpp,
                          double *L):
        cdef int ii, i
        cdef double tmp

        # First-derivative
        Lp[0] = 0
        # Second derivative
        Lpp[0] = 0
        # Objective value
        L[0] = 0

        for ii in xrange(n_nz):
            i = indices[ii]
            tmp = data[ii] * C
            Lpp[0] += data[ii] * tmp
            Lp[0] += b[i] * tmp
            L[0] += C * b[i] * b[i]

        Lpp[0] *= 2
        Lp[0] *= 2

    cdef void update(self,
                     int j,
                     double z_diff,
                     double C,
                     int *indices,
                     double *data,
                     int n_nz,
                     double *y,
                     double *b,
                     double *L_new):
        cdef int ii, i

        # New objective value
        L_new[0] = 0

        for ii in xrange(n_nz):
            i = indices[ii]
            # Update residuals.
            b[i] -= z_diff * data[ii]
            L_new[0] += C * b[i] * b[i]


cdef class SquaredHinge(LossFunction):

    def __init__(self,
                 int max_steps=20,
                 double sigma=0.01,
                 double beta=0.5,
                 int verbose=0):
        self.max_steps = max_steps
        self.sigma = sigma
        self.beta = beta
        self.verbose = verbose

    # Binary

    cdef void derivatives(self,
                          int j,
                          double C,
                          int *indices,
                          double *data,
                          int n_nz,
                          double *y,
                          double *b,
                          double *Lp,
                          double *Lpp,
                          double *L):
        cdef int i, ii
        cdef double val, tmp

        # First derivative
        Lp[0] = 0
        # Second derivative
        Lpp[0] = 0
        # Objective value
        L[0] = 0

        for ii in xrange(n_nz):
            i = indices[ii]
            val = data[ii] * y[i]

            if b[i] > 0:
                tmp = val * C
                Lp[0] -= b[i] * tmp
                Lpp[0] += val * tmp
                L[0] += C * b[i] * b[i]

        Lp[0] *= 2
        Lpp[0] *= 2

    cdef void update(self,
                     int j,
                     double z_diff,
                     double C,
                     int *indices,
                     double *data,
                     int n_nz,
                     double *y,
                     double *b,
                     double *L_new):
        cdef int i, ii
        cdef double b_new

        # New objective value
        L_new[0] = 0

        for ii in xrange(n_nz):
            i = indices[ii]
            b_new = b[i] + z_diff * data[ii] * y[i]
            # b[i] = 1 - y[i] * np.dot(w, X[i])
            b[i] = b_new
            if b_new > 0:
                L_new[0] += b_new * b_new

        L_new[0] *= C


    # Multiclass

    cdef void derivatives_mc(self,
                             int j,
                             double C,
                             int n_samples,
                             int n_vectors,
                             int* indices,
                             double *data,
                             int n_nz,
                             int* y,
                             double* b,
                             double* g,
                             double* h,
                             double* L,
                             double* Lpp_max):

        cdef int ii, i, k
        cdef double tmp, tmp2, b_val
        cdef double* b_ptr = b
        cdef double DBL_MAX = np.finfo(np.double).max

        # Largest second derivative.
        Lpp_max[0] = 0
        # Objective value
        L[0] = 0

        for k in xrange(n_vectors):
            # First derivative with respect to w_jk
            g[k] = 0
            # Second derivative with respect to  w_jk^2
            h[k] = 0

        for k in xrange(n_vectors):

            for ii in xrange(n_nz):
                i = indices[ii]

                if y[i] == k:
                    continue

                # b_val = b[k, i]
                b_val = b_ptr[i]

                if b_val > 0:
                    L[0] += C * b_val * b_val
                    tmp = C * data[ii]
                    tmp2 = tmp * b_val
                    g[y[i]] -= tmp2
                    g[k] += tmp2
                    tmp2 = tmp * data[ii]
                    h[y[i]] += tmp2
                    h[k] += tmp2

            b_ptr += n_samples

        Lpp_max[0] = -DBL_MAX
        for k in xrange(n_vectors):
            g[k] *= 2
            Lpp_max[0] = max(Lpp_max[0], h[k])

        Lpp_max[0] *= 2
        Lpp_max[0] = min(max(Lpp_max[0], LOWER), UPPER)

    cdef void update_mc(self,
                        double C,
                        int n_samples,
                        int n_vectors,
                        int* indices,
                        double *data,
                        int n_nz,
                        int* y,
                        double *b,
                        double *d,
                        double *d_old,
                        double* h,
                        double* L_new):

        cdef int ii, i, k
        cdef double tmp, b_new
        cdef double* b_ptr

        # New objective value
        L_new[0] = 0
        for ii in xrange(n_nz):
            i = indices[ii]
            b_ptr = b + i

            tmp = d_old[y[i]] - d[y[i]]

            for k in xrange(n_vectors):
                if k != y[i]:
                    # b_ptr[0] = b[k, i]
                    b_new = b_ptr[0] + (tmp - (d_old[k] - d[k])) * data[ii]
                    b_ptr[0] = b_new
                    if b_new > 0:
                        L_new[0] += C * b_new * b_new

                b_ptr += n_samples

    cdef void lipschitz_constant_mt(self,
                                    int n_vectors,
                                    ColumnDataset X,
                                    double C,
                                    double* out):

        cdef double scale = 2 * C * n_vectors
        self._lipschitz_constant(X, scale, out)

    cdef void lipschitz_constant_mc(self,
                                    int n_vectors,
                                    ColumnDataset X,
                                    double C,
                                    double* out):

        cdef double scale = 4 * C * (n_vectors - 1)
        self._lipschitz_constant(X, scale, out)


cdef class SmoothHinge(LossFunction):

    def __init__(self,
                 int max_steps=20,
                 double sigma=0.01,
                 double beta=0.5,
                 int verbose=0):
        self.max_steps = max_steps
        self.sigma = sigma
        self.beta = beta
        self.verbose = verbose

    cdef void derivatives(self,
                          int j,
                          double C,
                          int *indices,
                          double *data,
                          int n_nz,
                          double *y,
                          double *b,
                          double *Lp,
                          double *Lpp,
                          double *L):
        cdef int i, ii
        cdef double val, tmp
        cdef double gamma = 1.0

        # First derivative
        Lp[0] = 0
        # Second derivative
        Lpp[0] = 0
        # Objective value
        L[0] = 0

        for ii in xrange(n_nz):
            i = indices[ii]
            val = data[ii] * y[i]

            if 0 <= b[i] and b[i] <= 1:
                tmp = val * C
                Lp[0] -= b[i] * tmp / gamma
                Lpp[0] += val * tmp / gamma
                L[0] += C * 0.5 / gamma * b[i] * b[i]
            elif b[i] >= gamma:
                tmp = val * C
                Lp[0] -= tmp
                L[0] += C * (b[i] - 0.5 * gamma)


    cdef void update(self,
                     int j,
                     double z_diff,
                     double C,
                     int *indices,
                     double *data,
                     int n_nz,
                     double *y,
                     double *b,
                     double *L_new):
        cdef int i, ii
        cdef double b_new
        cdef double gamma = 1.0

        # New objective value
        L_new[0] = 0

        for ii in xrange(n_nz):
            i = indices[ii]
            b_new = b[i] + z_diff * data[ii] * y[i]
            # b[i] = 1 - y[i] * np.dot(w, X[i])
            b[i] = b_new

            if 0 <= b_new and b_new <= 1:
                L_new[0] += 0.5 / gamma * b_new * b_new
            elif b_new >= gamma:
                L_new[0] += b_new - 0.5 * gamma

        L_new[0] *= C


cdef class ModifiedHuber(LossFunction):

    def __init__(self,
                 int max_steps=30,
                 double sigma=0.01,
                 double beta=0.5,
                 int verbose=0):
        self.max_steps = max_steps
        self.sigma = sigma
        self.beta = beta
        self.verbose = verbose

    cdef void derivatives(self,
                          int j,
                          double C,
                          int *indices,
                          double *data,
                          int n_nz,
                          double *y,
                          double *b,
                          double *Lp,
                          double *Lpp,
                          double *L):
        cdef int i, ii
        cdef double val, tmp

        # First derivative
        Lp[0] = 0
        # Second derivative
        Lpp[0] = 0
        # Objective value
        L[0] = 0

        for ii in xrange(n_nz):
            i = indices[ii]
            val = data[ii] * y[i]

            if b[i] > 2:
                Lp[0] -= 2 * val * C
                # -4 yp = 4 (b[i] - 1)
                L[0] += 4 * C * (b[i] - 1)
            elif b[i] > 0:
                tmp = val * C
                Lp[0] -= b[i] * tmp
                Lpp[0] += val * tmp
                L[0] += C * b[i] * b[i]

        Lp[0] *= 2
        Lpp[0] *= 2

    cdef void update(self,
                     int j,
                     double z_diff,
                     double C,
                     int *indices,
                     double *data,
                     int n_nz,
                     double *y,
                     double *b,
                     double *L_new):
        cdef int i, ii
        cdef double b_new

        L_new[0] = 0

        for ii in xrange(n_nz):
            i = indices[ii]
            b_new = b[i] + z_diff * data[ii] * y[i]
            b[i] = b_new

            if b_new > 2:
                L_new[0] += 4 * C * (b[i] - 1)
            elif b_new > 0:
                L_new[0] += C * b_new * b_new


cdef class Log(LossFunction):

    def __init__(self,
                 int max_steps=30,
                 double sigma=0.01,
                 double beta=0.5,
                 int verbose=0):
        self.max_steps = max_steps
        self.sigma = sigma
        self.beta = beta
        self.verbose = verbose

    # Binary

    cdef void derivatives(self,
                          int j,
                          double C,
                          int *indices,
                          double *data,
                          int n_nz,
                          double *y,
                          double *b,
                          double *Lp,
                          double *Lpp,
                          double *L):
        cdef int i, ii
        cdef double val, tau, exppred, tmp

        # First derivative
        Lp[0] = 0
        # Second derivative
        Lpp[0] = 0
        # Objective value
        L[0] = 0

        for ii in xrange(n_nz):
            i = indices[ii]
            val = data[ii] * y[i]

            exppred = 1 + 1 / b[i]
            tau = 1 / exppred
            tmp = val * C
            Lp[0] += tmp * (tau - 1)
            Lpp[0] += tmp * val * tau * (1 - tau)
            L[0] += C * log(exppred)


    cdef void update(self,
                     int j,
                     double z_diff,
                     double C,
                     int *indices,
                     double *data,
                     int n_nz,
                     double *y,
                     double *b,
                     double *L_new):
        cdef int i, ii
        cdef double exppred

        # New objective value
        L_new[0] = 0

        for ii in xrange(n_nz):
            i = indices[ii]
            b[i] /= exp(z_diff * data[ii] * y[i])
            exppred = 1 + 1 / b[i]
            L_new[0] += C * log(exppred)

    # Multiclass

    cdef void derivatives_mc(self,
                             int j,
                             double C,
                             int n_samples,
                             int n_vectors,
                             int* indices,
                             double *data,
                             int n_nz,
                             int* y,
                             double* b,
                             double *g,
                             double* Z,
                             double* L,
                             double* Lpp_max):

        cdef int ii, i, k
        cdef double Lpp, tmp, tmp2
        cdef double* b_ptr
        cdef double DBL_MAX = np.finfo(np.double).max

        # Objective value
        L[0] = 0

        for ii in xrange(n_nz):
            i = indices[ii]
            b_ptr = b + i
            Z[i] = 0 # Normalization term
            for k in xrange(n_vectors):
                # b_ptr[0] = b[k, i]
                Z[i] += b_ptr[0]
                b_ptr += n_samples
            L[0] += C * log(Z[i])

        # Largest second derivative
        Lpp_max[0] = -DBL_MAX

        b_ptr = b
        for k in xrange(n_vectors):
            # First derivatives (k th element of the partial gradient)
            g[k] = 0
            # Second derivative
            Lpp = 0

            for ii in xrange(n_nz):
                i = indices[ii]

                if Z[i] == 0:
                    continue

                # b_ptr[i] = b[k, i]
                tmp = b_ptr[i] / Z[i]
                tmp2 = data[ii] * C
                Lpp += tmp2 * data[ii] * tmp * (1 - tmp)

                if k == y[i]:
                    tmp -= 1

                g[k] += tmp * tmp2

            Lpp_max[0] = max(Lpp, Lpp_max[0])
            b_ptr += n_samples

        Lpp_max[0] = min(max(Lpp_max[0], LOWER), UPPER)

    cdef void update_mc(self,
                        double C,
                        int n_samples,
                        int n_vectors,
                        int* indices,
                        double *data,
                        int n_nz,
                        int* y,
                        double* b,
                        double* d,
                        double* d_old,
                        double* Z,
                        double* L_new):
        cdef int i, ii, k
        cdef double tmp
        cdef double* b_ptr

        # New objective value
        L_new[0] = 0

        for ii in xrange(n_nz):
            i = indices[ii]
            b_ptr = b + i
            tmp = d_old[y[i]] - d[y[i]]
            Z[i] = 0

            for k in xrange(n_vectors):
                # b_ptr[0] = b[k, i]
                if y[i] != k:
                    b_ptr[0] *= exp((d[k] - d_old[k] + tmp) * data[ii])
                Z[i] += b_ptr[0]
                b_ptr += n_samples

            L_new[0] += C * log(Z[i])

    cdef void recompute(self,
                        ColumnDataset X,
                        double* y,
                        double* w,
                        double* b):
        cdef int n_samples = X.get_n_samples()
        cdef int n_features = X.get_n_features()
        cdef int i, j, ii
        cdef double tmp

        # Data pointers
        cdef double* data
        cdef int* indices
        cdef int n_nz

        for i in xrange(n_samples):
            b[i] = 0

        for j in xrange(n_features):
            X.get_column_ptr(j, &indices, &data, &n_nz)

            for ii in xrange(n_nz):
                i = indices[ii]
                b[i] += data[ii] * w[j]

        for i in xrange(n_samples):
            b[i] = exp(y[i] * b[i])

    cdef void recompute_mc(self,
                           int n_vectors,
                           ColumnDataset X,
                           int* y,
                           np.ndarray[double, ndim=2, mode='c'] w,
                           np.ndarray[double, ndim=2, mode='c'] b):
        cdef int n_samples = X.get_n_samples()
        cdef int n_features = X.get_n_features()
        cdef int i, j, k, k2, ii
        cdef double tmp

        # Data pointers
        cdef double* data
        cdef int* indices
        cdef int n_nz

        for i in xrange(n_samples):
            for k in xrange(n_vectors):
                b[k, i] = 0

        for j in xrange(n_features):
            X.get_column_ptr(j, &indices, &data, &n_nz)

            for ii in xrange(n_nz):
                i = indices[ii]

                for k in xrange(n_vectors):
                    tmp = w[k, j] * data[ii]
                    if k == y[i]:
                        for k2 in xrange(n_vectors):
                            if k2 != y[i]:
                                b[k2, i] -= tmp
                    else:
                        b[k, i] += tmp

        for i in xrange(n_samples):
            for k in xrange(n_vectors):
                if k != y[i]:
                    b[k, i] = exp(b[k, i])
                else:
                    b[k, i] = 1

    cdef void lipschitz_constant_mt(self,
                                    int n_vectors,
                                    ColumnDataset X,
                                    double C,
                                    double* out):

        cdef double scale = 0.25 * C * n_vectors
        self._lipschitz_constant(X, scale, out)

    cdef void lipschitz_constant_mc(self,
                                    int n_vectors,
                                    ColumnDataset X,
                                    double C,
                                    double* out):

        cdef double scale = C * 0.5
        self._lipschitz_constant(X, scale, out)


def _primal_cd(self,
               np.ndarray[double, ndim=2, mode='c'] w,
               np.ndarray[double, ndim=2, mode='c'] b,
               ColumnDataset X,
               np.ndarray[int, ndim=1] y,
               np.ndarray[double, ndim=2, mode='fortran'] Y,
               int k,
               int multiclass,
               np.ndarray[int, ndim=1, mode='c'] active_set,
               int penalty,
               LossFunction loss,
               selection,
               int permute,
               termination,
               double C,
               double alpha,
               int max_iter,
               int max_steps,
               int shrinking,
               double violation_init,
               RandomState rs,
               double tol,
               callback,
               int n_calls,
               int verbose):

    # Dataset
    cdef int n_samples = X.get_n_samples()
    cdef int n_features = X.get_n_features()
    cdef int n_vectors = w.shape[0]
    cdef int active_size = active_set.shape[0]
    cdef int active_size_start = active_size
    cdef double DBL_MAX = np.finfo(np.double).max

    # Counters
    cdef int t, s, i, j, n

    # Optimality violations
    cdef double violation_max_old = DBL_MAX
    cdef double violation_max
    cdef double violation
    cdef double violation_sum

    # Convergence
    cdef int check_violation_sum = termination == "violation_sum"
    cdef int check_violation_max = termination == "violation_max"
    cdef int stop = 0
    cdef int has_callback = callback is not None
    cdef int shrink = 0

    # Coordinate selection
    cdef int cyclic = selection == "cyclic"
    cdef int uniform = selection == "uniform"
    if uniform:
        permute = 0
        shrinking = 0

    # Lipschitz constants
    cdef np.ndarray[double, ndim=1, mode='c'] Lcst
    Lcst = np.zeros(n_features, dtype=np.float64)
    if max_steps == 0:
        if multiclass:
            loss.lipschitz_constant_mc(n_vectors, X, C, <double*>Lcst.data)
        else:
            loss.lipschitz_constant_mt(n_vectors, X, C, <double*>Lcst.data)

    # Vector containers
    cdef double* b_ptr
    cdef double* y_ptr
    cdef double* w_ptr
    cdef int* active_set_ptr = <int*>active_set.data
    cdef np.ndarray[double, ndim=1, mode='c'] g  # Partial gradient
    cdef np.ndarray[double, ndim=1, mode='c'] d  # Block update
    cdef np.ndarray[double, ndim=1, mode='c'] d_old  # Block update (old)
    cdef np.ndarray[double, ndim=1, mode='c'] buf  # Buffer
    cdef double* buf_ptr
    if k == -1:
        # Multiclass or multitask.
        g = np.zeros(n_vectors, dtype=np.float64)
        d = np.zeros(n_vectors, dtype=np.float64)
        d_old = np.zeros(n_vectors, dtype=np.float64)
        if multiclass:
            buf = np.zeros(n_samples, dtype=np.float64)
            buf_ptr = <double*>buf.data
        b_ptr = <double*>b.data
    else:
        # Binary classification or regression.
        b_ptr = <double*>b.data + k * n_samples
        y_ptr = <double*>Y.data + k * n_samples
        w_ptr = <double*>w.data + k * n_features
        buf = np.zeros(n_samples, dtype=np.float64)
        buf_ptr = <double*>buf.data

    for t in xrange(max_iter):
        # Permute features (cyclic case only)
        if permute:
            rs.shuffle(active_set[:active_size])

        # Initialize violations.
        violation_max = 0
        violation_sum = 0

        s = 0
        while s < active_size:
            # Select coordinate.
            if cyclic:
                j = active_set[s]
            elif uniform:
                j = active_set[rs.randint(active_size - 1)]

            # Solve sub-problem.
            if penalty == 1:
                shrink = loss.solve_l1(j, C, alpha, w_ptr, n_samples, X,
                                       y_ptr, b_ptr, Lcst[j], violation_max_old,
                                       &violation, shrinking)
            elif penalty == 12:
                shrink = loss.solve_l1l2(j, C, alpha, w, n_vectors, X,
                                         <int*>y.data, Y, multiclass,
                                         b, Lcst[j], <double*>g.data,
                                         <double*>d.data, <double*>d_old.data,
                                         buf_ptr, violation_max_old,
                                         &violation, shrinking)
            elif penalty == 2:
                loss.solve_l2(j, C, alpha, w_ptr, X, y_ptr, b_ptr, &violation)

            # Check if need to shrink.
            if shrink:
                active_size -= 1
                _swap(active_set_ptr, s, active_size)
                continue

            # Update violations.
            violation_max = max(violation_max, violation)
            violation_sum += violation

            # Callback
            if has_callback and s % n_calls == 0:
                ret = callback(self)
                if ret is not None:
                    stop = 1
                    break

            s += 1
        # end while active_size

        if stop:
            break

        # Initialize violations.
        if t == 0 and violation_init == 0:
            if check_violation_sum:
                violation_init = violation_sum
            elif check_violation_max:
                violation_init = violation_max

        # Verbose output.
        if verbose >= 1:
            if check_violation_sum:
                print("iter", t + 1, violation_sum / violation_init,
                      "(%d)" % active_size)
            elif check_violation_max:
                print("iter", t + 1, violation_max / violation_init,
                      "(%d)" % active_size)

        # Check convergence.
        if (check_violation_sum and
            violation_sum <= tol * violation_init) or \
           (check_violation_max and
            violation_max <= tol * violation_init):
            if active_size == active_size_start:
                if verbose >= 1:
                    print("\nConverged at iteration", t)
                break
            else:
                # When shrinking is enabled, we need to do one more outer
                # iteration on the entire optimization problem.
                active_size = active_size_start
                violation_max_old = DBL_MAX
                continue

        violation_max_old = violation_max

    if k == -1:
        return violation_init, w, b
    else:
        return violation_init, w[k], b[k]
