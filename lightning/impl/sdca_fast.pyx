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

ctypedef np.int64_t LONG

from libc.math cimport fabs

from lightning.impl.dataset_fast cimport RowDataset


cdef _add_l2(double* data,
             int* indices,
             int n_nz,
             double* w,
             double update,
             double* regul):

    cdef int j, jj
    cdef double delta, w_old

    for jj in range(n_nz):
        j = indices[jj]
        delta = update * data[jj]
        w_old = w[j]
        w[j] += delta
        regul[0] += delta * (2 * w_old + delta)


cdef inline double _truncate(double v,
                             double sigma):
    if v > sigma:
        return v - sigma
    elif v < -sigma:
        return v + sigma
    else:
        return 0


cdef _add_elastic(double* data,
                  int* indices,
                  int n_nz,
                  double*w,
                  double* v,
                  double update,
                  double* regul,
                  double sigma):

    cdef int j, jj
    cdef double delta, w_old, v_old

    for jj in range(n_nz):
        j = indices[jj]
        delta = update * data[jj]
        v_old = v[j]
        w_old = w[j]
        v[j] += delta
        w[j] = _truncate(v[j], sigma)
        regul[0] -= v_old * w_old
        regul[0] += v[j] * w[j]


cdef _sqnorms(RowDataset X,
              np.ndarray[double, ndim=1, mode='c'] sqnorms):

    cdef int n_samples = X.get_n_samples()
    cdef int i, j
    cdef double dot

    # Data pointers.
    cdef double* data
    cdef int* indices
    cdef int n_nz

    for i in range(n_samples):
        X.get_row_ptr(i, &indices, &data, &n_nz)
        dot = 0
        for jj in range(n_nz):
            dot += data[jj] * data[jj]
        sqnorms[i] = dot


cdef double _pred(double* data,
                  int* indices,
                  int n_nz,
                  double* w):

    cdef int j, jj
    cdef double dot = 0

    for jj in range(n_nz):
        j = indices[jj]
        dot += w[j] * data[jj]

    return dot


cdef _solve_subproblem(double*data,
                       int* indices,
                       int n_nz,
                       double y,
                       double* w,
                       double* v,
                       double* dcoef,
                       int loss_func,
                       double sqnorm,
                       double scale,
                       double sigma,
                       double gamma,
                       double* primal,
                       double* dual,
                       double* regul):

    cdef double pred, dcoef_old, residual, error, loss, update

    pred = _pred(data, indices, n_nz, w)

    dcoef_old = dcoef[0]

    if loss_func == 0:  # square loss
        residual = pred - y
        loss = 0.5 * residual * residual
        update = -(dcoef_old + residual) / (1 + sqnorm * scale)
        dual[0] += update * (y - dcoef_old - 0.5 * update)

    elif loss_func == 1:  # absolute loss
        residual = y - pred
        loss = fabs(residual)
        update = residual / (sqnorm * scale) + dcoef_old
        update = min(1.0, update)
        update = max(-1.0, update)
        update -= dcoef_old
        dual[0] += y * update

    elif loss_func == 2:  # hinge loss
        error = 1 - y * pred
        loss = max(0.0, error)
        update = error / (sqnorm * scale) + dcoef_old * y
        update = min(1.0, update)
        update = max(0.0, update)
        update = y * update - dcoef_old
        dual[0] += y * update

    elif loss_func == 3:  # smooth hinge loss
        error = 1 - y * pred

        if error < 0:
            loss = 0
        elif error > gamma:
            loss = error - 0.5 * gamma
        else:
            loss = 0.5 / gamma * error * error

        update = (error - gamma * dcoef_old * y) / (sqnorm * scale + gamma)
        update += dcoef_old * y
        update = min(1.0, update)
        update = max(0.0, update)
        update = y * update - dcoef_old
        dual[0] += y * update
        dual[0] -= gamma * dcoef_old * update
        dual[0] -= 0.5 * gamma * update * update

    elif loss_func == 4:  # squared hinge loss
        # Update is the same as squared loss but with a truncation.
        residual = pred - y
        update = -(dcoef_old + residual) / (1 + sqnorm * scale)
        if (dcoef_old + update) * y < 0:
            update = -dcoef_old

        error = 1 - y * pred
        if error >= 0:
            loss = residual * residual

        dual[0] += (y - dcoef_old) * update - 0.5 * update * update

    # Use accumulated loss rather than true primal objective value, which is
    # expensive to compute.
    primal[0] += loss

    if update != 0:
        dcoef[0] += update
        if sigma > 0:
            _add_elastic(data, indices, n_nz, w, v, update * scale, regul,
                         sigma)
        else:
            _add_l2(data, indices, n_nz, w, update * scale, regul)


def _prox_sdca_fit(self,
                   RowDataset X,
                   np.ndarray[double, ndim=1]y,
                   np.ndarray[double, ndim=1]coef,
                   np.ndarray[double, ndim=1]dual_coef,
                   double alpha1,
                   double alpha2,
                   int loss_func,
                   double gamma,
                   int max_iter,
                   double tol,
                   callback,
                   int n_calls,
                   int verbose,
                   rng):

    cdef int n_samples = X.get_n_samples()
    cdef int n_features = X.get_n_features()

    # Variables
    cdef double sigma, scale, primal, dual, regul, gap
    cdef int it, ii, i
    cdef int has_callback = callback is not None
    cdef LONG t

    # Pre-compute square norms.
    cdef np.ndarray[double, ndim=1, mode='c'] sqnorms
    sqnorms = np.zeros(n_samples, dtype=np.float64)
    _sqnorms(X, sqnorms)

    # Pointers
    cdef double* w = <double*>coef.data
    cdef double* dcoef = <double*>dual_coef.data
    cdef np.ndarray[double, ndim=1] v_data
    v_data = np.zeros(n_features, dtype=np.float64)
    cdef double* v = <double*>v_data.data
    cdef np.ndarray[int, ndim=1] sindices
    sindices = np.arange(n_samples, dtype=np.int32)

    # Data pointers.
    cdef double* data
    cdef int* indices
    cdef int n_nz

    if alpha1 > 0:  # Elastic-net case
        sigma = alpha1 / alpha2
    else:  # L2-only case
        sigma = 0

    scale = 1. / (alpha2 * n_samples)

    dual = 0
    regul = 0

    t = 0
    for it in range(max_iter):
        primal = 0

        rng.shuffle(sindices)

        for ii in range(n_samples):

            i = sindices[ii]

            if sqnorms[i] == 0:
                continue

            # Retrieve row.
            X.get_row_ptr(i, &indices, &data, &n_nz)

            _solve_subproblem(data, indices, n_nz, y[i], w, v, dcoef + i,
                              loss_func, sqnorms[i], scale, sigma, gamma,
                              &primal, &dual, &regul)

            if has_callback and t % n_calls == 0:
                ret = callback(self)
                if ret is not None:
                    break

            t += 1

        # end for ii in range(n_samples)

        gap = (primal - dual) / n_samples + alpha2 * regul
        gap = fabs(gap)

        if verbose:
            print("iter", it + 1, gap)

        if gap <= tol:
            if verbose:
                print("Converged")
            break

    # for it in range(max_iter)

    for i in range(n_samples):
        dcoef[i] *= scale
