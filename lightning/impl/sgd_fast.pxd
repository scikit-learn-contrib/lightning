# Author: Mathieu Blondel
# License: BSD

cdef class LossFunction:

    cpdef double loss(self, double p, double y)
    cpdef double get_update(self, double p, double y)


cdef class ModifiedHuber(LossFunction):

    cpdef double loss(self, double p, double y)

    cpdef double get_update(self, double p, double y)


cdef class Hinge(LossFunction):

    cdef double threshold

    def __init__(self, double threshold=1.0)
        self.threshold = threshold

    cpdef double loss(self, double p, double y)

    cpdef double get_update(self, double p, double y)


cdef class SmoothHinge(LossFunction):

    cdef double gamma

    def __init__(self, double gamma=1.0):
        self.gamma = gamma  # the larger, the smoother

    cpdef double loss(self, double p, double y)

    cpdef double get_update(self, double p, double y)


cdef class SquaredHinge(LossFunction):

    cdef double threshold

    def __init__(self, double threshold=1.0):
        self.threshold = threshold

    cpdef double loss(self, double p, double y)

    cpdef double get_update(self, double p, double y)


cdef class Log(LossFunction):

    cpdef double loss(self, double p, double y)

    cpdef double get_update(self, double p, double y)


cdef class SquaredLoss(LossFunction):

    cpdef double loss(self, double p, double y)

    cpdef double get_update(self, double p, double y)


cdef class Huber(LossFunction):

    cdef double c

    def __init__(self, double c):
        self.c = c

    cpdef double loss(self, double p, double y)

    cpdef double get_update(self, double p, double y)


cdef class EpsilonInsensitive(LossFunction):

    cdef double epsilon

    def __init__(self, double epsilon):
        self.epsilon = epsilon

    cpdef double loss(self, double p, double y)

    cpdef double get_update(self, double p, double y)

