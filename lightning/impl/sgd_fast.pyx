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

from lightning.impl.dataset_fast cimport RowDataset

ctypedef np.int64_t LONG

cdef extern from "math.h":
    cdef extern double fabs(double x)
    cdef extern double exp(double x)
    cdef extern double log(double x)
    cdef extern double sqrt(double x)
    cdef extern double pow(double x, double y)

cdef extern from "float.h":
   double DBL_MAX


cdef class LossFunction:

    cpdef double loss(self, double p, double y):
        raise NotImplementedError()

    cpdef double get_update(self, double p, double y):
        raise NotImplementedError()

cdef class ModifiedHuber(LossFunction):

    cpdef double loss(self, double p, double y):
        cdef double z = p * y
        if z >= 1.0:
            return 0.0
        elif z >= -1.0:
            return (1.0 - z) * (1.0 - z)
        else:
            return -4.0 * z

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

    cpdef double loss(self, double p, double y):
        cdef double z = p * y
        if z <= self.threshold:
            return (self.threshold - z)
        return 0.0

    cpdef double get_update(self, double p, double y):
        cdef double z = p * y
        if z <= self.threshold:
            return y
        return 0.0



cdef class Hinge01(LossFunction):

    cdef double threshold

    def __init__(self, double threshold=1.0):
        self.threshold = threshold

    cpdef double loss(self, double p, double y):
        cdef double z = self.threshold - 4 * (y - 0.5) * (p - 0.5)
        if z >= 0:
            return z
        return 0.0

    cpdef double get_update(self, double p, double y):
        cdef double z = self.threshold - 4 * (y - 0.5) * (p - 0.5)
        if z >= 0:
            return 4 * y - 2
        return 0.0


cdef class SquaredHinge(LossFunction):

    cdef double threshold

    def __init__(self, double threshold=1.0):
        self.threshold = threshold

    cpdef double loss(self, double p, double y):
        cdef double z = self.threshold - p * y
        if z > 0:
            return z * z
        return 0.0

    cpdef double get_update(self, double p, double y):
        cdef double z = self.threshold - p * y
        if z > 0:
            return 2 * y * z
        return 0.0


cdef class SquaredHinge01(LossFunction):

    cdef double threshold

    def __init__(self, double threshold=1.0):
        self.threshold = threshold

    cpdef double loss(self, double p, double y):
        cdef double z = self.threshold - 4 * (y - 0.5) * (p - 0.5)
        if z >= 0:
            return z * z
        return 0.0

    cpdef double get_update(self, double p, double y):
        cdef double z = self.threshold - 4 * (y - 0.5) * (p - 0.5)
        if z >= 0:
            return 2 * (4 * y - 2) * z
        return 0.0


cdef class Log(LossFunction):

    cpdef double loss(self, double p, double y):
        cdef double z = p * y
        # approximately equal and saves the computation of the log
        if z > 18:
            return exp(-z)
        if z < -18:
            return -z
        return log(1.0 + exp(-z))

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

    cpdef double loss(self, double p, double y):
        cdef double z = p * y
        # approximately equal and saves the computation of the log
        if z > self.threshold:
            return 0
        else:
            return log(1.0 + exp(-self.gamma * z))

    cpdef double get_update(self, double p, double y):
        cdef double z = p * y
        if z > self.threshold:
            return 0
        return self.gamma * y / (exp(self.gamma * z) + 1.0)

    cpdef double get_gamma(self):
        return self.gamma


cdef class SquaredLoss(LossFunction):

    cpdef double loss(self, double p, double y):
        return 0.5 * (p - y) * (p - y)

    cpdef double get_update(self, double p, double y):
        return y - p


cdef class Huber(LossFunction):

    cdef double c

    def __init__(self, double c):
        self.c = c

    cpdef double loss(self, double p, double y):
        cdef double r = p - y
        cdef double abs_r = abs(r)
        if abs_r <= self.c:
            return 0.5 * r * r
        else:
            return self.c * abs_r - (0.5 * self.c * self.c)

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

    cpdef double loss(self, double p, double y):
        cdef double ret = abs(y - p) - self.epsilon
        return ret if ret > 0 else 0

    cpdef double get_update(self, double p, double y):
        if y - p > self.epsilon:
            return 1
        elif p - y > self.epsilon:
            return -1
        else:
            return 0


cdef double _dot(np.ndarray[double, ndim=2, mode='c'] W,
                 int k,
                 int *indices,
                 double *data,
                 int n_nz):
    cdef int jj, j
    cdef double pred = 0.0

    for jj in xrange(n_nz):
        j = indices[jj]
        pred += data[jj] * W[k, j]

    return pred


cdef void _add(np.ndarray[double, ndim=2, mode='c'] W,
               int k,
               int *indices,
               double *data,
               int n_nz,
               double scale):
    cdef int jj, j

    for jj in xrange(n_nz):
        j = indices[jj]
        W[k, j] += data[jj] * scale


cdef double _get_eta(int learning_rate, double alpha,
                     double eta0, double power_t, LONG t):
    cdef double eta = eta0
    if learning_rate == 2: # PEGASOS
        eta = 1.0 / (alpha * t)
    elif learning_rate == 3: # INVERSE SCALING
        eta = eta0 / pow(t, power_t)
    return eta


cdef void _l1_update(double eta,
                     double alpha,
                     double* delta,
                     LONG* timestamps,
                     np.ndarray[double, ndim=2, mode='c'] W,
                     double* data,
                     int* indices,
                     int n_nz,
                     int k,
                     LONG t,
                     int non_negative):
    cdef int j, jj
    cdef double w_new
    cdef LONG tm1 = t - 1

    for jj in xrange(n_nz):
        j = indices[jj]

        if timestamps[j] == tm1:
            continue

        # delta[tm1] - delta[timestamps[j]] corresponds to the amount of
        # regularization buffered and that must applied before the weights can
        # be used to compute the current prediction.
        if non_negative:
            w_new = W[k, j] - (delta[tm1] - delta[timestamps[j]])
        else:
            w_new = fabs(W[k, j]) - (delta[tm1] - delta[timestamps[j]])

        if w_new <= 0:
            W[k, j] = 0
        elif W[k, j] > 0 or non_negative:
            W[k, j] = w_new
        else:
            W[k, j] = -w_new

        timestamps[j] = tm1

    delta[t] = delta[tm1] + eta * alpha


cdef void _nnl2_update(np.ndarray[double, ndim=2, mode='c'] W,
                       int* indices,
                       int n_nz,
                       int k):
    cdef int j, jj
    for jj in xrange(n_nz):
        j = indices[jj]
        if W[k, j] < 0:
            W[k, j] = 0


cdef void _l1_finalize(double* delta,
                       LONG* timestamps,
                       np.ndarray[double, ndim=2, mode='c'] W,
                       int k,
                       LONG t,
                       int non_negative):
    cdef int n_features = W.shape[1]
    cdef int j
    cdef double w_new

    for j in xrange(n_features):
        if timestamps[j] == t:
            continue

        if non_negative:
            w_new = W[k, j] - (delta[t] - delta[timestamps[j]])
        else:
            w_new = fabs(W[k, j]) - (delta[t] - delta[timestamps[j]])

        if w_new <= 0:
            W[k, j] = 0
        elif W[k, j] > 0 or non_negative:
            W[k, j] = w_new
        else:
            W[k, j] = -w_new


def _binary_sgd(self,
                np.ndarray[double, ndim=2, mode='c'] W,
                np.ndarray[double, ndim=1] intercepts,
                int k,
                RowDataset X,
                np.ndarray[double, ndim=1] y,
                LossFunction loss,
                int penalty,
                double alpha,
                int learning_rate,
                double eta0,
                double power_t,
                int fit_intercept,
                double intercept_decay,
                int max_iter,
                int shuffle,
                random_state,
                callback,
                int n_calls,
                int verbose):

    cdef Py_ssize_t n_samples = X.get_n_samples()
    cdef Py_ssize_t n_features = X.get_n_features()

    # Initialization
    cdef int i, ii
    cdef LONG t
    cdef double update, update_eta, update_eta_scaled, pred, eta, scale
    cdef double w_scale = 1.0
    cdef double intercept = 0.0
    cdef int has_callback = callback is not None
    cdef int nn_l1 = penalty == -1

    cdef np.ndarray[LONG, ndim=1, mode='c'] timestamps
    timestamps = np.zeros(n_features, dtype=np.int64)

    cdef np.ndarray[double, ndim=1, mode='c'] delta
    delta = np.zeros(max_iter + 1, dtype=np.float64)

    # Data pointers
    cdef double* data
    cdef int* indices
    cdef int n_nz

    # Training indices.
    cdef np.ndarray[int, ndim=1, mode='c'] index
    index = np.arange(n_samples, dtype=np.int32)

    for t in xrange(1, max_iter + 1):
        # Retrieve current training instance and shuffle if necessary.
        ii = (t-1) % n_samples
        if shuffle and ii == 0:
            random_state.shuffle(index)
        i = index[ii]
        eta = _get_eta(learning_rate, alpha, eta0, power_t, t)

        # Retrieve row.
        X.get_row_ptr(i, &indices, &data, &n_nz)

        if penalty == 1 or nn_l1: # L1-regularization.
            _l1_update(eta, alpha,
                       <double*>delta.data, <LONG*>timestamps.data,
                       W, data, indices, n_nz, k, t, nn_l1)

        # Compute current prediction.
        pred = _dot(W, k, indices, data, n_nz)
        pred *= w_scale
        pred += intercepts[k]

        update = loss.get_update(pred, y[i])

        # Update if necessary.
        if update != 0:
            update_eta = update * eta
            update_eta_scaled = update_eta / w_scale

            _add(W, k, indices, data, n_nz, update_eta_scaled)

            if fit_intercept:
                intercepts[k] += update_eta * intercept_decay

        if penalty == 2: # L2-regularization.
            w_scale *= (1 - alpha * eta)
        elif penalty == -2: # NN constraints + L2-regularization.
            w_scale *= 1 / (1 + alpha * eta)
            _nnl2_update(W, indices, n_nz, k)

        # Take care of possible underflow.
        if w_scale < 1e-9:
            W[k] *= w_scale
            w_scale = 1.0

        # Callback
        if has_callback and t % n_calls == 0:
            ret = callback(self)
            if ret is not None:
                break

    # Finalize.
    if penalty == 1 or nn_l1:
        _l1_finalize(<double*>delta.data, <LONG*>timestamps.data,
                     W, k, t, nn_l1)
    elif w_scale != 1.0:
        W[k] *= w_scale


cdef void _softmax(double* scores, int size):
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


cdef class MulticlassLossFunction:

    cdef void update(self,
                      double* scores,
                      int y,
                      double* data,
                      int* indices,
                      int n_nz,
                      int i,
                      np.ndarray[double, ndim=2, mode='c'] W,
                      double* w_scales,
                      double* intercepts,
                      double intercept_decay,
                      double eta,
                      int fit_intercept
                     ):
        raise NotImplementedError()


cdef class MulticlassHinge(MulticlassLossFunction):

    cdef void update(self,
                      double* scores,
                      int y,
                      double* data,
                      int* indices,
                      int n_nz,
                      int i,
                      np.ndarray[double, ndim=2, mode='c'] W,
                      double* w_scales,
                      double* intercepts,
                      double intercept_decay,
                      double eta,
                      int fit_intercept
                     ):
        cdef int n_vectors = W.shape[0]
        cdef double best = -DBL_MAX
        cdef int l, k

        for l in xrange(n_vectors):
            if scores[l] > best:
                best = scores[l]
                k = l

        # Update if necessary.
        if k != y:
            _add(W, k, indices, data, n_nz, -eta / w_scales[k])
            _add(W, y, indices, data, n_nz, eta / w_scales[y])

            if fit_intercept:
                scale = eta * intercept_decay
                intercepts[k] -= scale
                intercepts[y] += scale


cdef class MulticlassSquaredHinge(MulticlassLossFunction):

    cdef void update(self,
                      double* scores,
                      int y,
                      double* data,
                      int* indices,
                      int n_nz,
                      int i,
                      np.ndarray[double, ndim=2, mode='c'] W,
                      double* w_scales,
                      double* intercepts,
                      double intercept_decay,
                      double eta,
                      int fit_intercept
                     ):
        cdef int n_vectors = W.shape[0]
        cdef double u
        cdef int l

        for l in xrange(n_vectors):

            if y == l:
                continue

            u = 1 - scores[y] + scores[l]

            if u <= 0:
                continue

            u *= eta * 2

            _add(W, l, indices, data, n_nz, -u / w_scales[l])
            _add(W, y, indices, data, n_nz, u / w_scales[y])

            if fit_intercept:
                scale = u * intercept_decay
                intercepts[l] -= scale
                intercepts[y] += scale


cdef class MulticlassLog(MulticlassLossFunction):

    cdef void update(self,
                      double* scores,
                      int y,
                      double* data,
                      int* indices,
                      int n_nz,
                      int i,
                      np.ndarray[double, ndim=2, mode='c'] W,
                      double* w_scales,
                      double* intercepts,
                      double intercept_decay,
                      double eta,
                      int fit_intercept
                     ):
        cdef int n_vectors = W.shape[0]
        cdef double u
        cdef int l

        _softmax(scores, n_vectors)

        # Update.
        for l in xrange(n_vectors):
            if scores[l] != 0:
                if l == y:
                    # Need to update the correct label minus the prediction.
                    u = eta * (1 - scores[l])
                else:
                    # Need to update the incorrect label weighted by the
                    # prediction.
                    u = -eta * scores[l]

                _add(W, l, indices, data, n_nz, u / w_scales[l])

                if fit_intercept:
                    intercepts[l] += u * intercept_decay


cdef void _l1l2_update(double eta,
                       double alpha,
                       double* delta,
                       LONG* timestamps,
                       np.ndarray[double, ndim=2, mode='c'] W,
                       double* data,
                       int* indices,
                       int n_nz,
                       LONG t):
    cdef int j, jj, l
    cdef double scale, norm
    cdef int n_vectors = W.shape[0]
    cdef LONG tm1 = t - 1

    for jj in xrange(n_nz):
        j = indices[jj]

        if timestamps[j] == tm1:
            continue

        norm = 0
        for l in xrange(n_vectors):
            norm += W[l, j] * W[l, j]
        norm = sqrt(norm)

        scale = 1 - (delta[tm1] - delta[timestamps[j]]) / norm
        if scale < 0:
            scale = 0
        for l in xrange(n_vectors):
            W[l, j] *= scale

        timestamps[j] = tm1

    delta[t] = delta[tm1] + eta * alpha


cdef void _l1l2_finalize(double* delta,
                         LONG* timestamps,
                         np.ndarray[double, ndim=2, mode='c'] W,
                         LONG t):
    cdef int n_features = W.shape[1]
    cdef int n_vectors = W.shape[0]
    cdef double norm
    cdef int j, l

    for j in xrange(n_features):

        if timestamps[j] == t:
            continue

        norm = 0
        for l in xrange(n_vectors):
            norm += W[l, j] * W[l, j]
        norm = sqrt(norm)

        scale = 1 - (delta[t] - delta[timestamps[j]]) / norm
        if scale < 0:
            scale = 0
        for l in xrange(n_vectors):
            W[l, j] *= scale


def _multiclass_sgd(self,
                    np.ndarray[double, ndim=2, mode='c'] W,
                    np.ndarray[double, ndim=1] intercepts,
                    RowDataset X,
                    np.ndarray[int, ndim=1] y,
                    MulticlassLossFunction loss,
                    int penalty,
                    double alpha,
                    int learning_rate,
                    double eta0,
                    double power_t,
                    int fit_intercept,
                    double intercept_decay,
                    int max_iter,
                    int shuffle,
                    random_state,
                    callback,
                    int n_calls,
                    int verbose):

    cdef Py_ssize_t n_samples = X.get_n_samples()
    cdef Py_ssize_t n_features = X.get_n_features()
    cdef Py_ssize_t n_vectors = W.shape[0]

    # Initialization
    cdef int it, i, ii, l
    cdef LONG t
    cdef double pred, eta, scale, norm
    cdef double intercept = 0.0
    cdef np.ndarray[double, ndim=1, mode='c'] w_scales
    w_scales = np.ones(n_vectors, dtype=np.float64)
    cdef np.ndarray[double, ndim=1, mode='c'] scores
    scores = np.ones(n_vectors, dtype=np.float64)
    cdef int all_zero
    cdef int has_callback = callback is not None

    cdef np.ndarray[LONG, ndim=1, mode='c'] timestamps
    timestamps = np.zeros(n_features, dtype=np.int64)

    cdef np.ndarray[double, ndim=1, mode='c'] delta
    delta = np.zeros(max_iter + 1, dtype=np.float64)

    # Data pointers
    cdef double* data
    cdef int* indices
    cdef int n_nz

    # Training indices.
    cdef np.ndarray[int, ndim=1, mode='c'] index
    index = np.arange(n_samples, dtype=np.int32)

    for t in xrange(1, max_iter + 1):
        # Retrieve current training instance and shuffle if necessary.
        ii = (t-1) % n_samples
        if shuffle and ii == 0:
            random_state.shuffle(index)
        i = index[ii]
        eta = _get_eta(learning_rate, alpha, eta0, power_t, t)

        # Retrieve row.
        X.get_row_ptr(i, &indices, &data, &n_nz)

        # L1/L2 regularization.
        if penalty == 12:
            _l1l2_update(eta, alpha,
                         <double*>delta.data, <LONG*>timestamps.data,
                         W, data, indices, n_nz, t)

        # Compute predictions.
        for l in xrange(n_vectors):
            scores[l] = _dot(W, l, indices, data, n_nz)
            scores[l] *= w_scales[l]
            scores[l] += intercepts[l]

        # Update weight vectors.
        loss.update(<double*>scores.data, y[i],
                    data, indices, n_nz,
                    i, W, <double*>w_scales.data, <double*>intercepts.data,
                    intercept_decay, eta, fit_intercept)

        # L2 regularization.
        if penalty == 2:
            scale = (1 - alpha * eta)
            for l in xrange(n_vectors):
                w_scales[l] *= scale

                # Take care of possible underflow.
                if w_scales[l] < 1e-9:
                    W[l] *= w_scales[l]
                    w_scales[l] = 1.0

        # Callback
        if has_callback and t % n_calls == 0:
            ret = callback(self)
            if ret is not None:
                break

    # Finalize.
    if penalty == 2:
        for l in xrange(n_vectors):
            if w_scales[l] != 1.0:
                W[l] *= w_scales[l]
    elif penalty == 12:
        _l1l2_finalize(<double*>delta.data, <LONG*>timestamps.data,
                       W, t)

