# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: language_level=3
#
# Copyright 2005 Robert Kern (robert.kern@gmail.com)

cdef extern from "randomkit.h":

    ctypedef struct rk_state:
        unsigned long key[624]
        int pos
        int has_gauss
        double gauss

cdef class RandomState:

    cdef rk_state *internal_state
    cdef object initial_seed
    cpdef long randint(self, unsigned long high)
