# Copyright 2005 Robert Kern (robert.kern@gmail.com)

from libc cimport stdlib

import numpy as np
cimport numpy as np

cdef extern from "randomkit.h":

    ctypedef struct rk_state:
        unsigned long key[624]
        int pos
        int has_gauss
        double gauss

    ctypedef enum rk_error:
        RK_NOERR = 0
        RK_ENODEV = 1
        RK_ERR_MAX = 2

    char *rk_strerror[2]

    # 0xFFFFFFFFUL
    unsigned long RK_MAX

    void rk_seed(unsigned long seed, rk_state *state)
    rk_error rk_randomseed(rk_state *state)
    unsigned long rk_random(rk_state *state)
    long rk_long(rk_state *state)
    unsigned long rk_ulong(rk_state *state)
    unsigned long rk_interval(unsigned long max, rk_state *state)
    double rk_double(rk_state *state)
    void rk_fill(void *buffer, size_t size, rk_state *state)
    rk_error rk_devfill(void *buffer, size_t size, int strong)
    rk_error rk_altfill(void *buffer, size_t size, int strong,
            rk_state *state)
    double rk_gauss(rk_state *state)

cdef class RandomState:

    def __init__(self, seed=None):
        self.internal_state = <rk_state*>stdlib.malloc(sizeof(rk_state))

        self.seed(seed)

    def __dealloc__(self):
        if self.internal_state != NULL:
            stdlib.free(self.internal_state)
            self.internal_state = NULL

    def seed(self, seed=None):
        cdef rk_error errcode
        if seed is None:
            errcode = rk_randomseed(self.internal_state)
        elif type(seed) is int:
            rk_seed(seed, self.internal_state)
        elif isinstance(seed, np.integer):
            iseed = int(seed)
            rk_seed(iseed, self.internal_state)
        else:
            raise ValueError("Wrong seed")

    cpdef long randint(self, unsigned long high):
        return <long>rk_interval(high, self.internal_state)

    def shuffle(self, object x):
        cdef int i, j
        cdef int copy

        i = len(x) - 1
        try:
            j = len(x[0])
        except:
            j = 0

        if (j == 0):
            # adaptation of random.shuffle()
            while i > 0:
                j = rk_interval(i, self.internal_state)
                x[i], x[j] = x[j], x[i]
                i = i - 1
        else:
            # make copies
            copy = hasattr(x[0], 'copy')
            if copy:
                while(i > 0):
                    j = rk_interval(i, self.internal_state)
                    x[i], x[j] = x[j].copy(), x[i].copy()
                    i = i - 1
            else:
                while(i > 0):
                    j = rk_interval(i, self.internal_state)
                    x[i], x[j] = x[j][:], x[i][:]
                    i = i - 1
