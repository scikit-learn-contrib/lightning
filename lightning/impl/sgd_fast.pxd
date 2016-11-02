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

    cpdef double loss(self, double p, double y)

    cpdef double get_update(self, double p, double y)


cdef class SmoothHinge(LossFunction):

    cdef double gamma

    cpdef double loss(self, double p, double y)

    cpdef double get_update(self, double p, double y)


cdef class SquaredHinge(LossFunction):

    cdef double threshold

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

    cpdef double loss(self, double p, double y)

    cpdef double get_update(self, double p, double y)


cdef class EpsilonInsensitive(LossFunction):

    cdef double epsilon

    cpdef double loss(self, double p, double y)

    cpdef double get_update(self, double p, double y)

