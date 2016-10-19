import numpy as np
import scipy.sparse as sp

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.extmath import safe_sparse_dot


def _frank_wolfe(w_init, X, y, beta, max_iter=50, tol=1e-3, max_nz=None,
                verbose=0):
    """
    Solve

    0.5 * ||np.dot(X, w) - y||^2 s.t. ||w||_1 <= beta

    by the Frank-Wolfe method.

    The method can be seen as a greedy coordinate descent: it adds at most one
    non-zero coefficient per iteration.
    """
    n_samples, n_features = X.shape

    if sp.issparse(X):
        X = X.tocsc()

    w = w_init.copy()

    for it in range(max_iter):
        y_pred = safe_sparse_dot(X, w)
        resid = beta * y_pred - y
        neg_grad = -safe_sparse_dot(X.T, beta * resid)

        atom = np.argmax(np.abs(neg_grad))
        s = np.sign(neg_grad[atom])

        error = np.dot(resid, resid)
        dgap = s * neg_grad[atom] - np.dot(w, neg_grad)

        if it == 0:
            error_init = error
            dgap_init = dgap

        if verbose:
            print "iter", it + 1
            print "duality gap", dgap / dgap_init
            print "error reduction", error / error_init
            print "l1 norm", beta * np.sum(np.abs(w))
            print "n_nz", np.sum(w != 0)
            print

        # Find optimal step size by exact line search.
        Xs = s * X[:, atom]
        if sp.issparse(Xs):
            Xs_sq = np.dot(Xs.data, Xs.data)
        else:
            Xs_sq = np.dot(Xs, Xs)
        y_pred_sq = np.dot(y_pred, y_pred)
        b = (Xs - y_pred)
        gamma = np.dot(resid, y_pred) - safe_sparse_dot(resid, Xs)
        gamma /= beta * (Xs_sq - 2 * safe_sparse_dot(Xs.T, y_pred) + y_pred_sq)
        gamma = max(0, min(1, gamma))

        # Update parameters.
        w *= (1 - gamma)
        w[atom] += gamma * s

        # Stop if maximum number of non-zero coefficients is reached.
        if max_nz is not None and np.sum(w != 0) == max_nz:
            break

        # Stop if desired duality gap tolerance is reached.
        if dgap / dgap_init <= tol:
            if verbose:
                print "Converged"
            break

    w *= beta
    return w


class FWRegressor(BaseEstimator, RegressorMixin):

    def __init__(self, beta=1.0, max_iter=50, tol=1e-3, max_nz=None, verbose=0):
        self.beta = beta
        self.max_iter = max_iter
        self.tol = tol
        self.max_nz = max_nz
        self.verbose = verbose

    def fit(self, X, y):
        n_features = X.shape[1]
        coef = np.zeros(n_features)
        self.coef_ = _frank_wolfe(coef, X, y, beta=self.beta,
                                  max_iter=self.max_iter, tol=self.tol,
                                  max_nz=self.max_nz, verbose=self.verbose)
        return self

    def predict(self, X):
        return safe_sparse_dot(X, self.coef_)


if __name__ == '__main__':
    from sklearn.datasets import load_diabetes
    from sklearn.preprocessing import StandardScaler

    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
    X = StandardScaler().fit_transform(X)
    #X = sp.csr_matrix(X)

    reg = FWRegressor(beta=100, max_iter=1000, tol=1e-2, verbose=1)
    reg.fit(X, y)
    y_pred = reg.predict(X)
    error = np.mean((y - y_pred) ** 2)
    print error
