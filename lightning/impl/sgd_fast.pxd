# Author: Mathieu Blondel
# License: BSD

cdef class LossFunction:

    # return the loss, where p is the predicted value
    # y is the true value and i is the index of a sample
    # (used only for the case of weighted samples)
    cpdef double loss(self, double p, double y, int i)
    cpdef double get_update(self, double p, double y, int i)
