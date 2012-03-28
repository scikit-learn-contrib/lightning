cimport numpy as np

from lightning.kernel_fast cimport Kernel

cpdef decision_function_alpha(np.ndarray[double, ndim=2, mode='c'] X,
                              np.ndarray[double, ndim=2, mode='c'] sv,
                              np.ndarray[double, ndim=2, mode='c'] alpha,
                              np.ndarray[double, ndim=1, mode='c'] b,
                              Kernel kernel,
                              np.ndarray[double, ndim=2, mode='c'] out)

cpdef predict_alpha(np.ndarray[double, ndim=2, mode='c'] X,
                    np.ndarray[double, ndim=2, mode='c'] sv,
                    np.ndarray[double, ndim=2, mode='c'] alpha,
                    np.ndarray[double, ndim=1, mode='c'] b,
                    np.ndarray[int, ndim=1, mode='c'] classes,
                    Kernel kernel,
                    np.ndarray[double, ndim=1, mode='c'] out)
