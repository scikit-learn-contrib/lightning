
cdef c_prox_tv1d(double* w, size_t width, size_t incr, double stepsize)

cdef c_prox_tv2d(double* x, size_t n_rows, size_t n_cols, double stepsize, int max_iter, double tol)