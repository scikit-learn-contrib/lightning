import numpy as np
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


def prox_sdca_fit(X, y, alpha, l1_ratio, loss_func, max_iter, tol,
                  random_state=None):
    rs = check_random_state(random_state)
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

        rs.shuffle(indices)

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

    return v, dual_coef


if __name__ == '__main__':
    from sklearn.datasets import make_classification
    from sklearn.datasets import load_iris
    from sklearn.svm import LinearSVC

    X, y = make_classification(n_samples=200, n_features=50, n_classes=2,
                               random_state=0)
    y = y * 2 - 1

    #iris = load_iris()
    #X, y = iris.data, iris.target
    #X = X[y <= 1]
    #y = y[y <= 1]
    #y *= 2
    #y -= 1

    print LinearSVC(C=1.0).fit(X, y).score(X, y)

    w, dual_coef = prox_sdca_fit(X, y, alpha=X.shape[0], l1_ratio=0,
                                 loss_func=2, max_iter=50, tol=1e-3,
                                 random_state=0)

    y_pred = np.sign(np.dot(X, w))
    print np.mean(y == y_pred)
