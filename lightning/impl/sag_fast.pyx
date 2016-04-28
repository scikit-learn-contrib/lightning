# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
#
# Authors: Mathieu Blondel
#          Fabian Pedregosa
#          Arnaud Rachez
# License: BSD

import numpy as np
cimport numpy as np

ctypedef np.int64_t LONG

from libc.math cimport sqrt, fabs

from lightning.impl.randomkit.random_fast cimport RandomState
from lightning.impl.dataset_fast cimport RowDataset
from lightning.impl.sgd_fast cimport LossFunction

# Reimplementation for MSVC support
cdef inline double fmax(double a, double b) nogil:
    return max(a, b)


cdef class Penalty:

    cdef void projection(self,
                         double* w,
                         int* indices,
                         double stepsize,
                         int n_nz):
        raise NotImplementedError()
    
    cdef void projection_lagged(self,
                                int t,
                                double* w,
                                double* g_sum,
                                int* indices,
                                double stepsize_prox,
                                double stepsize_grad,
                                double* lag_scaling,
                                int n_nz,
                                int* last,
                                double* scaling_seq):
        raise NotImplementedError()

    cdef double regularization(self, np.ndarray[double, ndim=1] coef):
        raise NotImplementedError()


cdef class L1Penalty(Penalty):

    def __cinit__(self):
        self.support_lagged = True

    cdef void projection(self,
                         double* w,
                         int* indices,
                         double stepsize,
                         int n_nz):

        cdef int j, jj

        for jj in range(n_nz):
            j = indices[jj]
            w[j] = fmax(w[j] - stepsize, 0) - fmax(-w[j] - stepsize, 0)

    cdef void projection_lagged(self,
                                int t,
                                double* w,
                                double* g_sum,
                                int* indices,
                                double stepsize_prox,
                                double stepsize_grad,
                                double* geom_sum,
                                int n_nz,
                                int* last,
                                double* scaling_seq):
        cdef int i, j, jj
        cdef long missed_updates
        cdef double tmp

        for jj in range(n_nz):
            ind = indices[jj]
            missed_updates = t - last[ind]
            if missed_updates == 0:
                continue
            if fabs(g_sum[ind]) <= stepsize_prox:
                tmp = scaling_seq[t-1] / scaling_seq[last[ind]]
                scaling = geom_sum[t-1] - geom_sum[last[ind]] * tmp

                w[ind] -= stepsize_grad * scaling * g_sum[ind]
                w[ind] = fmax(w[ind] - scaling * stepsize_prox, 0) \
                        - fmax(-w[ind] - scaling * stepsize_prox, 0)
            else:
                for i in range(missed_updates, 0, -1):
                    w[ind] -= scaling_seq[i-1] * stepsize_grad * g_sum[ind]
                    tmp = stepsize_prox * scaling_seq[i-1]
                    w[ind] = fmax(w[ind] - tmp, 0) - fmax(-w[ind] - tmp, 0)
            last[ind] = t
        return

    cdef double regularization(self, np.ndarray[double, ndim=1] coef):

        cdef int j
        cdef double reg = 0

        for j in range(coef.size):
            reg += fabs(coef[j])
        return reg


cdef double _pred(double* data,
                  int* indices,
                  int n_nz,
                  double* w):

    cdef int j, jj
    cdef double dot = 0

    for jj in xrange(n_nz):
        j = indices[jj]
        dot += w[j] * data[jj]

    return dot

cdef void _add(double* data,
               int* indices,
               int n_nz,
               double scale,
               double* w):
    cdef int jj, j

    for jj in xrange(n_nz):
        j = indices[jj]
        w[j] += scale * data[jj]


cdef void _lagged_update(int t,
                         double* w,
                         double* g_sum,
                         double* geom_sum,
                         double* scaling_seq,
                         int* indices,
                         int n_nz,
                         int* last,
                         double stepsize):
    """
    Apply missing updates to w, just-in-time. See [1, Section 4] 
    for a description of this technique.

    [1] 1. Schmidt, M., Roux, N. Le & Bach, F. Minimizing Finite 
        Sums with the Stochastic Average Gradient. 1–45 (2013).
    """
    cdef long missed_updates
    cdef double scaling
    cdef double tmp

    for jj in range(n_nz):
        ind = indices[jj]
        missed_updates = t - last[ind]
        if missed_updates == 0:
            continue
        elif missed_updates == 1:
            scaling = 0.
        else:
            tmp = scaling_seq[t-1] / scaling_seq[last[ind]]
            scaling = geom_sum[t-1] - geom_sum[last[ind]] * tmp
        w[ind] -= stepsize * (1 + scaling) * g_sum[ind]
        last[ind] = t


def _sag_fit(self,
             RowDataset X,
             np.ndarray[double, ndim=1]y,
             np.ndarray[double, ndim=1]coef,
             np.ndarray[double, ndim=1]coef_scale,
             np.ndarray[double, ndim=1]grad,
             np.ndarray[double, ndim=1]sample_weight,
             double eta,
             double alpha,
             double beta,
             LossFunction loss,
             Penalty penalty,
             int max_iter,
             int n_inner,
             double tol,
             int verbose,
             callback,
             RandomState rng,
             bint saga,
             int adaptive_eta):

    cdef int n_samples = X.get_n_samples()
    cdef int n_features = X.get_n_features()

    # Variables.
    cdef int i, jj, j, it, t
    cdef double y_pred, scale, g_old, tmp, alpha_scaled
    cdef double violation, violation_init, violation_ratio
    cdef double eta_alpha = eta * alpha
    if eta_alpha == 1.:
        # this is a problem because then w_scale[0]
        # becomes zero. Solution: decrease slightly eta
        eta = 0.9 * eta
        eta_alpha = eta * alpha

    cdef double eta_avg = eta / n_samples
    cdef double g_change = 0.
    cdef int has_callback = callback is not None

    # Data pointers.
    cdef double* data
    cdef int* indices
    cdef int n_nz

    # Buffers and pointers.
    cdef np.ndarray[int, ndim=1]last_ = np.zeros(n_features, dtype=np.int32)
    cdef np.ndarray[int, ndim=1] last_penalty_ = np.zeros(n_features, dtype=np.int32)
    cdef np.ndarray[double, ndim=1] g_sum_
    cdef np.ndarray[int, ndim=1] all_indices_ = np.arange(n_features, dtype=np.int32)
    g_sum_ = np.zeros(n_features, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] lag_scaling_
    lag_scaling_ = np.empty(n_inner+2, dtype=np.float64)
    cdef np.ndarray[double, ndim=1] scaling_seq_
    scaling_seq_ = np.empty(n_inner+1, dtype=np.float64)
    cdef double* scaling_seq = <double*> scaling_seq_.data

    cdef np.ndarray[double, ndim=1] w_violation_
    cdef double* w_violation
    cdef double* g_sum = <double*>g_sum_.data
    cdef double* w = <double*>coef.data
    cdef double* w_scale = <double*>coef_scale.data
    
    cdef double* g = <double*>grad.data
    cdef double* geom_sum = <double*> lag_scaling_.data

    cdef int* last = <int*> last_.data
    cdef int* last_penalty_update = <int*> last_penalty_.data
    cdef int* all_indices = <int*> all_indices_.data
    cdef double geosum = 1.0
    cdef bint support_lagged = True
    cdef bint nontrivial_prox = saga and (penalty is not None)
    cdef double square_norm_i = 0.0
    cdef int line_search_freq = 10  # frequency of line search
    # Lipschitz constant of the loss terms. Start with the double
    # of the initial Lipschitz constant
    cdef double lipschitz
    cdef double line_search_scaling
    cdef np.ndarray[double, ndim=1] square_norm_X
    if adaptive_eta > 0:
        lipschitz = 1.0 / eta - alpha
        # the scaling is chosen so that the stepsize doubles if the
        # line search condition is verified for a whole pass over the data
        line_search_scaling = 2.0 ** (float(line_search_freq) / n_inner)
        square_norm_X = np.zeros(n_inner, dtype=np.float64)

    if nontrivial_prox:
        w_violation_ = np.zeros(n_features, dtype=np.float64)
        w_violation = <double*>w_violation_.data
        support_lagged = penalty.support_lagged

    geom_sum[0] = 0.0
    scaling_seq[0] = 1.0

    # Initialize gradient memory.
    for i in range(n_samples):
        # Retrieve sample i.
        X.get_row_ptr(i, &indices, &data, &n_nz)

        # Make prediction.
        y_pred = _pred(data, indices, n_nz, w) * w_scale[0]

        # A gradient is given by g[i] * X[i].
        g[i] = -sample_weight[i] * loss.get_update(y_pred, y[i])

        # Update g_sum.
        _add(data, indices, n_nz, g[i], g_sum)


    # Outer loop.
    for it in range(max_iter):

        # Inner loop.
        for t in range(n_inner):
            i = rng.randint(n_samples - 1)

            # Retrieve sample i.
            X.get_row_ptr(i, &indices, &data, &n_nz)

            # Apply missed updates.
            if t > 0 and support_lagged:
                if nontrivial_prox:
                    # SAGA with non-trivial prox
                    penalty.projection_lagged(
                        t, w, g_sum, indices, beta * eta / w_scale[0],
                        eta_avg / w_scale[0], geom_sum, n_nz, last,
                        scaling_seq)
                else:
                    # SAG or SAGA with trivial prox
                    _lagged_update(t, w, g_sum, geom_sum, scaling_seq,
                                   indices, n_nz, last, eta_avg / w_scale[0])

            # Make prediction.
            y_pred = _pred(data, indices, n_nz, w) * w_scale[0]

            # Make copy of old gradient value.
            g_old = g[i]

            # A gradient is given by g[i] * X[i].
            g[i] = - sample_weight[i] * loss.get_update(y_pred, y[i])
            g_change = g[i] - g_old


            # line-search procedure, to be done every 5 iterations
            # for details, see section 4.6 of Schmidt, M., Roux, N., & Bach, F. (2013).
            # "Minimizing finite sums with the stochastic average gradient".
            # arXiv Preprint arXiv:1309.2388
            if adaptive_eta > 0 and t % line_search_freq == 0 and fabs(g[i]) > 1e-8:
                if square_norm_X[i] == 0:
                    # compute the norm if not done already
                    for j in range(n_nz):
                        square_norm_X[i] += (data[j] * data[j])
                a = sample_weight[i] * loss.loss(y_pred - g[i] * square_norm_X[i] / lipschitz, y[i])
                b = sample_weight[i] * loss.loss(y_pred, y[i]) - 0.5 * square_norm_X[i] * (g[i] * g[i]) / lipschitz
                if a <= b :
                    # condition is satisfied, decrease Lipschitz constant
                    lipschitz /= line_search_scaling
                else:
                    # condition not satisfied, decrease step size
                    lipschitz *= 2.0

                # update eta_alpha and related
                eta = 1.0 / (lipschitz + alpha)
                eta_avg = eta / n_samples
                eta_alpha = eta * alpha

            if w_scale[0] < 1e-10:
                # bring coordinates up to date
                if support_lagged:
                    if nontrivial_prox:
                        penalty.projection_lagged(
                            t, w, g_sum, all_indices, beta * eta / w_scale[0],
                            eta_avg / w_scale[0], geom_sum, n_features, last,
                            scaling_seq)
                    else:
                        _lagged_update(t, w, g_sum, geom_sum, scaling_seq,
                                       all_indices, n_features, last, eta_avg / w_scale[0])
                # rescale features
                for j in range(n_features):
                    w[j] *= w_scale[0]
                w_scale[0] = 1.0
                geom_sum[t] = 0.0
                scaling_seq[t] = w_scale[0]

            # Update coefficient scale (l2 regularization).
            w_scale[0] *= (1 - eta_alpha)
            scaling_seq[t+1] = w_scale[0]
            geom_sum[t+1] = (1 + geom_sum[t]) * (1 - eta_alpha)

            if saga:
                # update w with sparse step bit
                _add(data, indices, n_nz, -g_change * eta / w_scale[0], w)

                if support_lagged:
                    # gradient-average part of the step
                    _lagged_update(t + 1, w, g_sum, geom_sum, scaling_seq,
                                   indices, n_nz, last, eta_avg / w_scale[0])
                    if nontrivial_prox:
                        # prox update
                        penalty.projection(w, indices, beta * eta / w_scale[0],
                                           n_nz)
                else:
                    # gradient-average part of the step
                    # could be an _add instead of a _lagged update since we are not
                    # using the last array anywhere else
                    _lagged_update(t + 1, w, g_sum, geom_sum, scaling_seq,
                                   all_indices, n_features, last, eta_avg / w_scale[0])
                    if nontrivial_prox:
                        # prox update
                        penalty.projection(w, all_indices, beta * eta / w_scale[0],
                                           n_features)

            # Update g_sum.
            _add(data, indices, n_nz, g_change, g_sum)


        # Finalize.
        if support_lagged:
            if nontrivial_prox:
                penalty.projection_lagged(
                    n_inner, w, g_sum, all_indices, beta * eta / w_scale[0],
                    eta_avg / w_scale[0], geom_sum, n_features, last,
                    scaling_seq)
            else:
                _lagged_update(n_inner, w, g_sum, geom_sum, scaling_seq,
                               all_indices, n_features, last, eta_avg / w_scale[0])

        for j in range(n_features):
            w[j] *= w_scale[0]
            last[j] = 0
            last_penalty_update[j] = 0
        w_scale[0] = 1.0

        # Callback.
        if has_callback:
            ret = callback(self)
            if ret is not None:
                break

        # Compute optimality violation.
        violation = 0
        alpha_scaled = alpha * w_scale[0]
        if nontrivial_prox:
            for j in range(n_features):
                    w_violation[j] = w_scale[0] * w[j] - \
                            eta * (g_sum[j] / n_samples + alpha_scaled * w[j])

            penalty.projection(w_violation, all_indices, beta * eta / w_scale[0],
                               n_features)

            for j in range(n_features):
                violation += (w_scale[0] * w[j] - w_violation[j])**2

        else:
            for j in range(n_features):
                tmp = g_sum[j] / n_samples + alpha_scaled * w[j]
                violation += tmp * tmp

        # Convergence monitoring.
        if it == 0:
            if violation != 0:
                violation_init = violation
            else:
                # First epoch is optimal.  Setting violation_init to a positive
                # value to avoid division by zero.
                violation_init = 1.0

        violation_ratio = violation / violation_init

        if verbose:
            print(it + 1, violation_ratio)

        if violation_ratio <= tol:
            if verbose:
                print("Converged")
            break

    # Rescale coefficients.
    for j in range(n_features):
        w[j] *= w_scale[0]
    w_scale[0] = 1.0
