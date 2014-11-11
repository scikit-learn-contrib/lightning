import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state



def _add_l2(Xi, w, update, regul):
    for j in xrange(Xi.shape[0]):
        delta = update * Xi[j]
        w_old = w[j]
        w[j] += delta
        regul[0] += delta * (2 * w_old + delta)


def _truncate(v, sigma):
    if v > sigma:
        return v - sigma
    elif v < -sigma:
        return v + sigma
    else:
        0


def _add_elastic(Xi, w, v, update, regul, sigma):
    for j in xrange(Xi.shape[0]):
        delta = update * Xi[j]
        v_old = v[j]
        w_old = w[j]
        v[j] += delta
        w[j] = _truncate(v[j], sigma)
        regul[0] -= v_old * w_old
        regul[0] += v[j] * w[j]


def prox_sdca_fit(X, y, alpha, l1_ratio, loss_func, max_iter, tol, rng):
    n_samples = X.shape[0]
    n_features = X.shape[1]

    if l1_ratio > 0:
        alpha = alpha * (1 - l1_ratio)
        sigma = l1_ratio / (1 - l1_ratio)

    scale = 1. / (alpha * n_samples)

    dual = 0
    regul = np.zeros(1, dtype=np.float64)

    sqnorms = (X ** 2).sum(axis=1)
    dual_coef = np.zeros(n_samples, dtype=np.float64)
    w = np.zeros(n_features, dtype=np.float64)
    v = np.zeros(n_features, dtype=np.float64)

    indices = np.arange(n_samples, dtype=np.int32)

    for it in xrange(max_iter):
        primal = 0

        rng.shuffle(indices)

        for ii in xrange(n_samples):

            i = indices[ii]

            if sqnorms[i] == 0:
                continue

            if l1_ratio > 0:  # elasticnet case
                pred = np.dot(X[i], w)
            else:
                pred = np.dot(X[i], v)

            dcoef_old = dual_coef[i]

            if loss_func == 0:  # square loss
                residual = pred - y[i]
                loss = 0.5 * residual * residual
                update = -(dcoef_old + residual) / (1 + sqnorms[i] * scale)
                dual += update * (y[i] - dcoef_old - 0.5 * update)

            elif loss_func == 1:  # absolute loss
                residual = y[i] - pred
                loss = abs(residual)
                update = residual / (sqnorms[i] * scale) + dcoef_old
                update = min(1.0, update)
                update = max(-1.0, update)
                update -= dcoef_old
                dual += y[i] * update

            elif loss_func == 2:  # hinge loss
                margin = 1 - y[i] * pred
                loss = max(0.0, margin)
                update = margin / (sqnorms[i] * scale) + dcoef_old * y[i]
                update = min(1.0, update)
                update = max(0.0, update)
                update = y[i] * update - dcoef_old
                dual += y[i] * update

            primal += loss

            if update != 0:
                dual_coef[i] += update
                if l1_ratio > 0:
                    _add_elastic(X[i], w, v, update * scale, regul, sigma)
                else:
                    _add_l2(X[i], v, update * scale, regul)

        # end for ii in xrange(n_samples)

        gap = (primal - dual) / n_samples + alpha * regul[0]
        print gap
        if gap <= tol:
            break

    # for it in xrange(max_iter)

    dual_coef *= scale

    return v


class ProxSDCA_Classifier(BaseEstimator, ClassifierMixin):

    def __init__(self, alpha=1.0, l1_ratio=0, loss="hinge", max_iter=10,
                 tol=1e-3, random_state=None):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.loss = loss
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def _get_loss(self):
        losses = {
            "squared": 0,
            "absolute": 1,
            "hinge": 2,
        }
        return losses[self.loss]

    def fit(self, X, y):
        rng = check_random_state(self.random_state)
        loss = self._get_loss()
        self.coef_ = prox_sdca_fit(X, y, self.alpha, self.l1_ratio, loss,
                                   self.max_iter, self.tol, rng)
        return self

    def predict(self, X):
        return np.sign(np.dot(X, self.coef_))
