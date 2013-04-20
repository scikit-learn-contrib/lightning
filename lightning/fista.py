# Author: Mathieu Blondel
# License: BSD

import numpy as np

from sklearn.base import ClassifierMixin
from sklearn.utils.extmath import safe_sparse_dot

from .base import BaseClassifier

from .dataset_fast import get_dataset

from .loss_fast import SquaredHinge
from .loss_fast import MulticlassSquaredHinge

from .penalty import NNConstraint
from .penalty import L1Penalty
from .penalty import L1L2Penalty


class FistaClassifier(BaseClassifier, ClassifierMixin):

    def __init__(self, C=1.0, alpha=1.0,
                 loss="squared_hinge", penalty="l1", multiclass=False,
                 max_iter=100,
                 max_steps=30, eta=2.0, sigma=1e-5,
                 callback=None, verbose=0):
        self.C = C
        self.alpha = alpha
        self.loss = loss
        self.penalty = penalty
        self.multiclass = multiclass
        self.max_iter = max_iter
        self.max_steps = max_steps
        self.eta = eta
        self.sigma = 1e-5
        self.callback = callback
        self.verbose = verbose

    def _get_loss(self):
        if self.multiclass:
            losses = {
                "squared_hinge": MulticlassSquaredHinge(),
            }
        else:
            losses = {
                "squared_hinge": SquaredHinge(),
            }

        return losses[self.loss]

    def _get_penalty(self):
        penalties = {
            "nn": NNConstraint(),
            "l1": L1Penalty(),
            "l1/l2": L1L2Penalty(),
        }
        return penalties[self.penalty]

    def _get_objective(self, df, y, Y, loss, penalty, coef):
        if self.multiclass:
            obj = self.C * loss.objective(df, y)
        else:
            obj = self.C * loss.objective(df, Y)
        obj += self.alpha * penalty.regularization(coef)
        return obj

    def _get_quad_approx(self, coefa, coefb, gradb, dfb, y, Y, L,
                         loss, penalty):
        if self.multiclass:
            approx = self.C * loss.objective(dfb, y)
        else:
            approx = self.C * loss.objective(dfb, Y)
        diff = coefa - coefb
        approx += np.sum(diff * gradb)
        approx += L / 2 * np.sum(diff ** 2)
        approx += self.alpha * penalty.regularization(coefa)
        return approx

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y, n_classes, n_vectors = self._set_label_transformers(y,
                                                               reencode=True)
        Y = np.asfortranarray(self.label_binarizer_.transform(y),
                              dtype=np.float64)

        loss = self._get_loss()
        penalty = self._get_penalty()
        ds = get_dataset(X)

        df = np.zeros((n_samples, n_vectors), dtype=np.float64)
        coef = np.zeros((n_vectors, n_features), dtype=np.float64)
        coefx = coef
        G = np.zeros((n_vectors, n_features), dtype=np.float64)

        obj = self._get_objective(df, y, Y, loss, penalty, coef)

        if self.max_steps == 0:
            # No line search, need to use constant step size.
            L = self.C * loss.max_gradient(ds, n_vectors)
        else:
            # Do not bother to compute the Lipschitz constant (expensive).
            L = 1.0

        t = 1.0
        for it in xrange(self.max_iter):
            if self.verbose >= 1:
                print "Iter", it + 1

            # Save current values
            t_old = t
            coefx_old = coefx

            # Gradient
            G.fill(0.0)
            if self.multiclass:
                loss.gradient(df, ds, y, G)
            else:
                loss.gradient(df, ds, Y, G)
            G *= self.C

            # Line search
            for tt in xrange(self.max_steps):
                # Solve
                coefx = coef - G / L
                coefx = penalty.projection(coefx, self.alpha, L)

                dfx = safe_sparse_dot(X, coefx.T)
                obj = self._get_objective(dfx, y, Y, loss, penalty, coefx)
                approx = self._get_quad_approx(coefx, coef, G, df, y, Y, L,
                                               loss, penalty)

                accepted = obj <= approx

                # Sufficient decrease condition
                if accepted:
                    if self.verbose >= 2:
                        print "Accepted at", tt + 1
                    break
                else:
                    L *= self.eta

            if self.max_steps == 0:
                coefx = coef - G / L
                coefx = penalty.projection(coefx, self.alpha, L)

            t = (1 + np.sqrt(1 + 4 * t_old * t_old) / 2)
            coef = coefx + (t_old - 1) / t * (coefx - coefx_old)
            df = safe_sparse_dot(X, coef.T)

            # Callback might need self.coef_.
            self.coef_ = coef
            if self.callback is not None:
                ret = self.callback(self)
                if ret is not None:
                    break

        return self
