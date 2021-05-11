# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: language_level=3
#
# Authors: Fabian Pedregosa
# License: BSD

cimport numpy as np
from cython cimport floating

cpdef prox_tv1d(np.ndarray[ndim=1, dtype=floating] w, floating stepsize)
