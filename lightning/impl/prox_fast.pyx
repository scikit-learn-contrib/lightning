# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
#
# Authors: Fabian Pedregosa
#

"""
These are some helper functions to compute the proximal operator of some common penalties
"""

import numpy as np
cimport numpy as np
from libc.math cimport fabs

def prox_tv1d(np.ndarray[ndim=1, dtype=double] w, double stepsize):
    """
    Computes the proximal operator of the 1-dimensional total variation operator.

    This solves a problem of the form

         argmin_x TV(x) + (1/(2 stepsize)) ||x - w||^2

    where TV(x) is the one-dimensional total variation

    Parameters
    ----------
    w: array
        vector of coefficients
    stepsize: float
        step size (sometimes denoted gamma) in proximal objective function

    References
    ----------
    Condat, Laurent. "A direct algorithm for 1D total variation denoising."
    IEEE Signal Processing Letters (2013)
    """
    cdef np.ndarray[ndim=1, dtype=double] tmp = w.copy()
    c_prox_tv1d(<double *> tmp.data, <size_t> w.size, 1, stepsize)
    return tmp

cdef c_prox_tv1d(double* w, size_t width, size_t incr, double stepsize):
    cdef long k, k0, kplus, kminus
    cdef double umin, umax, vmin, vmax, twolambda, minlambda

    # /to avoid invalid memory access to input[0] and invalid lambda values
    if width > 0 and stepsize >= 0:
        k, k0 = 0, 0			# k: current sample location, k0: beginning of current segment
        umin = stepsize  # u is the dual variable
        umax = - stepsize
        vmin = w[0] - stepsize
        vmax = w[0] + stepsize	  # bounds for the segment's value
        kplus = 0
        kminus = 0 	# last positions where umax=-lambda, umin=lambda, respectively
        twolambda = 2.0 * stepsize  # auxiliary variable
        minlambda = -stepsize		# auxiliary variable
        while True:				# simple loop, the exit test is inside
            while k >= width-1: 	# we use the right boundary condition
                if umin < 0.0:			# vmin is too high -> negative jump necessary
                    while True:
                        w[incr * k0] = vmin
                        k0 += 1
                        if k0 > kminus:
                            break
                    k = k0
                    kminus = k
                    vmin = w[incr * kminus]
                    umin = stepsize
                    umax = vmin + umin - vmax
                elif umax > 0.0:    # vmax is too low -> positive jump necessary
                    while True:
                        w[incr * k0] = vmax
                        k0 += 1
                        if k0 > kplus:
                            break
                    k = k0
                    kplus = k
                    vmax = w[incr * kplus]
                    umax = minlambda
                    umin = vmax + umax -vmin
                else:
                    vmin += umin / (k-k0+1)
                    while True:
                        w[incr * k0] = vmin
                        k0 += 1
                        if k0 > k:
                            break
                    return
            umin += w[incr * (k + 1)] - vmin
            if umin < minlambda:       # negative jump necessary
                while True:
                    w[incr * k0] = vmin
                    k0 += 1
                    if k0 > kminus:
                        break
                k = k0
                kminus = k
                kplus = kminus
                vmin = w[incr * kplus]
                vmax = vmin + twolambda
                umin = stepsize
                umax = minlambda
            else:
                umax += w[incr * (k + 1)] - vmax
                if umax > stepsize:
                    while True:
                        w[incr * k0] = vmax
                        k0 += 1
                        if k0 > kplus:
                            break
                    k = k0
                    kminus = k
                    kplus = kminus
                    vmax = w[incr * kplus]
                    vmin = vmax - twolambda
                    umin = stepsize
                    umax = minlambda
                else:                   # no jump necessary, we continue
                    k += 1
                    if umin >= stepsize:		# update of vmin
                        kminus = k
                        vmin += (umin - stepsize) / (kminus - k0 + 1)
                        umin = stepsize
                    if umax <= minlambda:	    # update of vmax
                        kplus = k
                        vmax += (umax + stepsize) / (kplus - k0 + 1)
                        umax = minlambda


cdef inline void prox_col_pass(double* a, size_t n_rows,
                               size_t n_cols, double stepsize):
    cdef size_t i
    for i in range(n_cols):
        c_prox_tv1d(a + i, n_rows, n_cols, stepsize)


cdef inline void prox_row_pass(double* a, size_t n_rows,
                             size_t n_cols, double stepsize):
    cdef size_t i
    for i in range(n_rows):
        c_prox_tv1d(a + i * n_cols, n_cols, 1, stepsize)


cdef c_prox_tv2d(double* x, size_t n_rows, size_t n_cols,
                double stepsize, int max_iter, double tol):
    """
    Douflas-Rachford to minimize a 2-dimensional total variation.

    Reference: https://arxiv.org/abs/1411.0589
    """
    cdef np.ndarray[ndim=2, dtype=double] p_ = np.zeros((n_rows, n_cols))
    cdef np.ndarray[ndim=2, dtype=double] q_ = np.zeros((n_rows, n_cols))
    cdef np.ndarray[ndim=2, dtype=double] y_ = np.zeros((n_rows, n_cols))
    cdef double* y = <double*> y_.data
    cdef double* p = <double*> p_.data
    cdef double* q = <double*> q_.data
    cdef size_t it, i, j, size = n_rows * n_cols

    # set X to the content of x
    for it in range(max_iter):
        for i in range(size):
            y[i] = x[i] + p[i]
        prox_col_pass(y, n_rows, n_cols, stepsize)
        for i in range(size):
            p[i] += x[i] - y[i]
            x[i] = y[i] + q[i]
        prox_row_pass(x, n_rows, n_cols, stepsize)
        for i in range(size):
            q[i] = y[i] + q[i] - x[i]

        # check convergence
        for i in range(size):
            if fabs(q[i] - x[i]) < tol:
                continue
            else:
                break
        else:
            # if all coordinates are below tol
            return


def prox_tv2d(np.ndarray[ndim=2, dtype=double] w, double stepsize,
              max_iter=500, tol=1e-3):
    """
    Computes the proximal operator of the 2-dimensional total variation operator.

    This solves a problem of the form

         argmin_x TV(x) + (1/(2 stepsize)) ||x - w||^2

    where TV(x) is the two-dimensional total variation. It does so using the
    Douglas-Rachford algorithm [Barbero and Sra, 2014].

    Parameters
    ----------
    w: array
        vector of coefficients

    stepsize: float
        step size (often denoted gamma) in proximal objective function

    max_iter: int

    tol: float

    References
    ----------
    Condat, Laurent. "A direct algorithm for 1D total variation denoising."
    IEEE Signal Processing Letters (2013)

    Barbero, Álvaro, and Suvrit Sra. "Modular proximal optimization for
    multidimensional total-variation regularization." arXiv preprint
    arXiv:1411.0589 (2014).
    """

    cdef np.ndarray[ndim=2, dtype=double] x = w.copy()
    c_prox_tv2d(<double*> x.data, w.shape[0], w.shape[1],
                stepsize, max_iter, tol)
    return x