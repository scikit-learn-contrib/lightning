# Author: Mathieu Blondel
# License: BSD

cdef class LossFunction:

    cpdef double loss(self, double p, double y)
    cpdef double get_update(self, double p, double y)
