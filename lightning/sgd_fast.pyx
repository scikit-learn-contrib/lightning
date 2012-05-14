# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
#
# Author: Mathieu Blondel
#         Peter Prettenhofer (loss functions)
# License: BSD

import numpy as np
cimport numpy as np

from cython.operator cimport dereference as deref
from cython.operator cimport preincrement as inc
from cython.operator cimport predecrement as dec

from libcpp.list cimport list

from lightning.kernel_fast cimport KernelCache
from lightning.kernel_fast cimport Kernel

cdef extern from "math.h":
    cdef extern double exp(double x)
    cdef extern double log(double x)
    cdef extern double sqrt(double x)
    cdef extern double pow(double x, double y)

cdef extern from "float.h":
   double DBL_MAX

cdef class LossFunction:

    cpdef double get_update(self, double p, double y):
        raise NotImplementedError()


cdef class ModifiedHuber(LossFunction):

    cpdef double get_update(self, double p, double y):
        cdef double z = p * y
        if z >= 1.0:
            return 0.0
        elif z >= -1.0:
            return 2.0 * (1.0 - z) * y
        else:
            return 4.0 * y


cdef class Hinge(LossFunction):

    cdef double threshold

    def __init__(self, double threshold=1.0):
        self.threshold = threshold

    cpdef double get_update(self, double p, double y):
        cdef double z = p * y
        if z <= self.threshold:
            return y
        return 0.0


cdef class Log(LossFunction):

    cpdef double get_update(self, double p, double y):
        cdef double z = p * y
        # approximately equal and saves the computation of the log
        if z > 18.0:
            return exp(-z) * y
        if z < -18.0:
            return y
        return y / (exp(z) + 1.0)


cdef class SparseLog(LossFunction):

    cdef double threshold, gamma

    def __init__(self, double threshold=0.99):
        self.threshold = threshold
        self.gamma = -log((1 - threshold)/threshold)

    cpdef double get_update(self, double p, double y):
        cdef double z = p * y
        if z > self.threshold:
            return 0
        return self.gamma * y / (exp(self.gamma * z) + 1.0)

    cpdef double get_gamma(self):
        return self.gamma


cdef class SquaredLoss(LossFunction):

    cpdef double get_update(self, double p, double y):
        return y - p


cdef class Huber(LossFunction):

    cdef double c

    def __init__(self, double c):
        self.c = c

    cpdef double get_update(self, double p, double y):
        cdef double r = p - y
        cdef double abs_r = abs(r)
        if abs_r <= self.c:
            return -r
        elif r > 0.0:
            return -self.c
        else:
            return self.c


cdef class EpsilonInsensitive(LossFunction):

    cdef double epsilon

    def __init__(self, double epsilon):
        self.epsilon = epsilon

    cpdef double get_update(self, double p, double y):
        if y - p > self.epsilon:
            return 1
        elif p - y > self.epsilon:
            return -1
        else:
            return 0


cdef double _dot(np.ndarray[double, ndim=2, mode='c'] W,
                 int k,
                 np.ndarray[double, ndim=2, mode='c'] X,
                 int i):
    cdef Py_ssize_t n_features = X.shape[1]
    cdef int j
    cdef double pred = 0.0

    for j in xrange(n_features):
        pred += X[i, j] * W[k, j]

    return pred


cdef double _kernel_dot(np.ndarray[double, ndim=2, mode='c'] W,
                        int k,
                        np.ndarray[double, ndim=2, mode='c'] X,
                        int i,
                        KernelCache kcache,
                        np.ndarray[double, ndim=1, mode='c'] col):
    cdef int j
    cdef double pred = 0
    cdef list[int].iterator it

    kcache.compute_column_sv(X, X, i, col)
    it = kcache.support_set.begin()

    while it != kcache.support_set.end():
        j = deref(it)
        pred += col[j] * W[k, j]
        inc(it)

    return pred


cdef void _add(np.ndarray[double, ndim=2, mode='c'] W,
               int k,
               np.ndarray[double, ndim=2, mode='c'] X,
               int i,
               double scale):
    cdef Py_ssize_t n_features = X.shape[1]
    cdef int j

    for j in xrange(n_features):
        W[k, j] += X[i, j] * scale


cdef double _get_eta(int learning_rate, double lmbda,
                     double eta0, double power_t, long t):
    cdef double eta = eta0
    if learning_rate == 2: # PEGASOS
        eta = 1.0 / (lmbda * t)
    elif learning_rate == 3: # INVERSE SCALING
        eta = eta0 / pow(t, power_t)
    return eta


def _binary_sgd(self,
                np.ndarray[double, ndim=2, mode='c'] W,
                np.ndarray[double, ndim=1] intercepts,
                int k,
                np.ndarray[double, ndim=2, mode='c'] X,
                np.ndarray[double, ndim=1] y,
                LossFunction loss,
                KernelCache kcache,
                int linear_kernel,
                int model_size,
                double lmbda,
                int learning_rate,
                double eta0,
                double power_t,
                int fit_intercept,
                double intercept_decay,
                int max_iter,
                random_state,
                int verbose):

    cdef Py_ssize_t n_samples = X.shape[0]
    cdef Py_ssize_t n_features = X.shape[1]

    cdef np.ndarray[int, ndim=1, mode='c'] indices
    indices = np.arange(n_samples, dtype=np.int32)

    cdef int n, i
    cdef long t = 1
    cdef double update, update_eta, update_eta_scaled, pred, eta
    cdef double w_scale = 1.0
    cdef double intercept = 0.0

    cdef np.ndarray[double, ndim=1, mode='c'] col
    if not linear_kernel:
        col = np.zeros(n_samples, dtype=np.float64)

    random_state.shuffle(indices)

    for t in xrange(1, max_iter + 1):
        i = indices[(t-1) % n_samples]

        if linear_kernel:
            pred = _dot(W, k, X, i)
        else:
            pred = _kernel_dot(W, k, X, i, kcache, col)

        pred *= w_scale
        pred += intercepts[k]

        eta = _get_eta(learning_rate, lmbda, eta0, power_t, t)
        update = loss.get_update(pred, y[i])

        if update != 0:
            update_eta = update * eta
            update_eta_scaled = update_eta / w_scale

            if linear_kernel:
                _add(W, k, X, i, update_eta_scaled)
            else:
                W[k, i] += update_eta_scaled

            if fit_intercept:
                intercepts[k] += update_eta * intercept_decay

        w_scale *= (1 - lmbda * eta)

        if w_scale < 1e-9:
            W[k] *= w_scale
            w_scale = 1.0

        # Update support vector set.
        if not linear_kernel:
            if W[k, i] == 0:
                kcache.remove_sv(i)
            else:
                kcache.add_sv(i)

        # Stop if necessary.
        if model_size > 0 and kcache.n_sv() >= model_size:
            break

    if w_scale != 1.0:
        W[k] *= w_scale


cdef int _predict_multiclass(np.ndarray[double, ndim=2, mode='c'] W,
                             np.ndarray[double, ndim=1] w_scales,
                             np.ndarray[double, ndim=1] intercepts,
                             np.ndarray[double, ndim=2, mode='c'] X,
                             int i):
    cdef Py_ssize_t n_features = X.shape[1]
    cdef Py_ssize_t n_vectors = W.shape[0]
    cdef int j, l

    cdef double pred
    cdef double best = -DBL_MAX
    cdef int selected = 0

    for l in xrange(n_vectors):
        pred = 0

        for j in xrange(n_features):
            pred += X[i, j] * W[l, j]

        pred *= w_scales[l]
        pred += intercepts[l]

        # pred += loss(y_true, y_pred)

        if pred > best:
            best = pred
            selected = l

    return selected


cdef int _kernel_predict_multiclass(np.ndarray[double, ndim=2, mode='c'] W,
                                    np.ndarray[double, ndim=1] w_scales,
                                    np.ndarray[double, ndim=1] intercepts,
                                    np.ndarray[double, ndim=2, mode='c'] X,
                                    int i,
                                    KernelCache kcache,
                                    np.ndarray[double, ndim=1] col):
    cdef Py_ssize_t n_features = X.shape[1]
    cdef Py_ssize_t n_vectors = W.shape[0]
    cdef int j, l
    cdef double pred
    cdef double best = -DBL_MAX
    cdef int selected = 0
    cdef list[int].iterator it

    kcache.compute_column_sv(X, X, i, col)

    for l in xrange(n_vectors):
        pred = 0

        it = kcache.support_set.begin()
        while it != kcache.support_set.end():
            j = deref(it)
            pred += col[j] * W[l, j]
            inc(it)

        pred *= w_scales[l]
        pred += intercepts[l]

        # pred += loss(y_true, y_pred)

        if pred > best:
            best = pred
            selected = l

    return selected


def _multiclass_hinge_sgd(self,
                          np.ndarray[double, ndim=2, mode='c'] W,
                          np.ndarray[double, ndim=1] intercepts,
                          np.ndarray[double, ndim=2, mode='c'] X,
                          np.ndarray[int, ndim=1] y,
                          KernelCache kcache,
                          int linear_kernel,
                          int model_size,
                          double lmbda,
                          int learning_rate,
                          double eta0,
                          double power_t,
                          int fit_intercept,
                          double intercept_decay,
                          int max_iter,
                          random_state,
                          int verbose):

    cdef Py_ssize_t n_samples = X.shape[0]
    cdef Py_ssize_t n_features = X.shape[1]
    cdef Py_ssize_t n_vectors = W.shape[0]

    cdef np.ndarray[int, ndim=1, mode='c'] indices
    indices = np.arange(n_samples, dtype=np.int32)

    cdef int it, i, l
    cdef long t = 1
    cdef double update, pred, eta, scale
    cdef double intercept = 0.0

    cdef np.ndarray[double, ndim=1, mode='c'] w_scales
    w_scales = np.ones(n_vectors, dtype=np.float64)

    cdef np.ndarray[double, ndim=1, mode='c'] col
    if not linear_kernel:
        col = np.zeros(n_samples, dtype=np.float64)

    random_state.shuffle(indices)

    for t in xrange(t, max_iter + 1):
        i = indices[(t-1) % n_samples]

        eta = _get_eta(learning_rate, lmbda, eta0, power_t, t)

        if linear_kernel:
            k = _predict_multiclass(W, w_scales, intercepts, X, i)
        else:
            k = _kernel_predict_multiclass(W, w_scales, intercepts, X, i,
                                           kcache, col)

        if k != y[i]:
            if linear_kernel:
                _add(W, k, X, i, -eta / w_scales[k])
                _add(W, y[i], X, i, eta / w_scales[y[i]])
            else:
                W[k, i] -= eta / w_scales[k]
                W[y[i], i] += eta / w_scales[y[i]]

            if fit_intercept:
                scale = eta * intercept_decay
                intercepts[k] -= scale
                intercepts[y[i]] += scale


        scale = (1 - lmbda * eta)
        for l in xrange(n_vectors):
            w_scales[l] *= scale

            if w_scales[l] < 1e-9:
                W[l] *= w_scales[l]
                w_scales[l] = 1.0

        # Update support vector set.
        if not linear_kernel:
            if W[k, i] == 0 and W[y[i], i] == 0:
                kcache.remove_sv(i)
            else:
                kcache.add_sv(i)

        # Stop if necessary.
        if model_size > 0 and kcache.n_sv() >= model_size:
            break

    for l in xrange(n_vectors):
        if w_scales[l] != 1.0:
            W[l] *= w_scales[l]


cdef void _softmax(np.ndarray[double, ndim=1] scores):
    cdef Py_ssize_t size = scores.shape[0]
    cdef double sum_ = 0
    cdef double max_score = -DBL_MAX
    cdef int i

    for i in xrange(size):
        max_score = max(max_score, scores[i])

    for i in xrange(size):
        scores[i] -= max_score
        if scores[i] < -10:
            scores[i] = 0
        else:
            scores[i] = exp(scores[i])
            sum_ += scores[i]

    if sum_ > 0:
        for i in xrange(size):
            scores[i] /= sum_


def _multiclass_log_sgd(self,
                        np.ndarray[double, ndim=2, mode='c'] W,
                        np.ndarray[double, ndim=1] intercepts,
                        np.ndarray[double, ndim=2, mode='c'] X,
                        np.ndarray[int, ndim=1] y,
                        KernelCache kcache,
                        int linear_kernel,
                        int model_size,
                        double lmbda,
                        int learning_rate,
                        double eta0,
                        double power_t,
                        int fit_intercept,
                        double intercept_decay,
                        int max_iter,
                        random_state,
                        int verbose):

    cdef Py_ssize_t n_samples = X.shape[0]
    cdef Py_ssize_t n_features = X.shape[1]
    cdef Py_ssize_t n_vectors = W.shape[0]

    cdef np.ndarray[int, ndim=1, mode='c'] indices
    indices = np.arange(n_samples, dtype=np.int32)

    cdef int it, i, l
    cdef long t = 1
    cdef double update, pred, eta, scale
    cdef double intercept = 0.0

    cdef np.ndarray[double, ndim=1, mode='c'] w_scales
    w_scales = np.ones(n_vectors, dtype=np.float64)

    cdef np.ndarray[double, ndim=1, mode='c'] scores
    scores = np.ones(n_vectors, dtype=np.float64)

    cdef np.ndarray[double, ndim=1, mode='c'] col
    if not linear_kernel:
        col = np.zeros(n_samples, dtype=np.float64)

    cdef int all_zero

    random_state.shuffle(indices)

    for t in xrange(t, max_iter + 1):
        i = indices[(t-1) % n_samples]

        eta = _get_eta(learning_rate, lmbda, eta0, power_t, t)

        for l in xrange(n_vectors):
            if linear_kernel:
                scores[l] = _dot(W, l, X, i)
            else:
                scores[l] = _kernel_dot(W, l, X, i, kcache, col)

            scores[l] *= w_scales[l]
            scores[l] += intercepts[l]

        _softmax(scores)

        for l in xrange(n_vectors):
            if scores[l] != 0:
                if l == y[i]:
                    # Need to update the correct label minus the prediction.
                    update = eta * (1 - scores[l])
                else:
                    # Need to update the incorrect label weighted by the
                    # prediction.
                    update = -eta * scores[l]

                if linear_kernel:
                    _add(W, l, X, i, update / w_scales[l])
                else:
                    W[l, i] += update / w_scales[l]

                if fit_intercept:
                    intercepts[l] += update * intercept_decay

        scale = (1 - lmbda * eta)
        for l in xrange(n_vectors):
            w_scales[l] *= scale

            if w_scales[l] < 1e-9:
                W[l] *= w_scales[l]
                w_scales[l] = 1.0

        # Update support vector set.
        if not linear_kernel:
            all_zero = 1
            for l in xrange(n_vectors):
                if W[l, i] != 0:
                    all_zero = 0
                    break
            if all_zero:
                kcache.remove_sv(i)
            else:
                kcache.add_sv(i)

        # Stop if necessary.
        if model_size > 0 and kcache.n_sv() >= model_size:
            break

    for l in xrange(n_vectors):
        if w_scales[l] != 1.0:
            W[l] *= w_scales[l]
