# Author: Mathieu Blondel
# License: BSD

import numpy as np

from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_random_state
from sklearn.utils.extmath import safe_sparse_dot

from .base import BaseClassifier

from .dataset_fast import get_dataset

from .loss_fast import SquaredHinge
from .loss_fast import MulticlassSquaredHinge

from .penalty import L1Penalty
from .penalty import L1L2Penalty
from .penalty import NNConstraint


class SparsaClassifier(BaseClassifier, ClassifierMixin):

    def __init__(self, C=1.0, alpha=1.0,
                 loss="squared_hinge", penalty="l1", multiclass=False,
                 max_iter=100,
                 Lmin=1e-30, Lmax=1e30, L_factor=0.8,
                 max_steps=30, eta=2.0, sigma=1e-5,
                 callback=None, verbose=0):
        self.C = C
        self.alpha = alpha
        self.loss = loss
        self.penalty = penalty
        self.multiclass = multiclass
        self.max_iter = max_iter
        self.Lmin = Lmin
        self.Lmax = Lmax
        self.L_factor = L_factor
        self.max_steps = max_steps
        self.eta = eta
        self.sigma = 1e-5
        self.callback = callback
        self.verbose = verbose

    def _get_loss(self):
        if self.multiclass:
            losses = {
                "squared_hinge" : MulticlassSquaredHinge(),
            }
        else:
            losses = {
                "squared_hinge" : SquaredHinge(),
            }

        return losses[self.loss]

    def _get_penalty(self):
        penalties = {
            "l1" : L1Penalty(),
            "l1/l2" : L1L2Penalty(),
        }
        return penalties[self.penalty]

    def _get_objective(self, df, y, Y, loss, penalty, coef):
        if self.multiclass:
            obj = self.C * loss.objective(df, y)
        else:
            obj = self.C * loss.objective(df, Y)
        obj += self.alpha * penalty.regularization(coef)
        return obj

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y, n_classes, n_vectors = \
                self._set_label_transformers(y, reencode=True)
        Y = np.asfortranarray(self.label_binarizer_.transform(y),
                              dtype=np.float64)

        loss = self._get_loss()
        penalty = self._get_penalty()
        ds = get_dataset(X)

        df = np.zeros((n_samples, n_vectors), dtype=np.float64)
        coef = np.zeros((n_vectors, n_features), dtype=np.float64)
        G = np.zeros((n_vectors, n_features), dtype=np.float64)

        obj = self._get_objective(df, y, Y, loss, penalty, coef)

        L = 1.0
        for t in xrange(self.max_iter):
            if self.verbose >= 1:
                print "Iter", t + 1

            # Save current values
            coef_old = coef
            obj_old = obj

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
                coef = coef_old - G / L
                coef = penalty.projection(coef, self.alpha, L)

                # New objective value
                df = safe_sparse_dot(X, coef.T)
                obj = self._get_objective(df, y, Y, loss, penalty, coef)

                # Difference with previous iteration
                s = coef - coef_old
                ss = np.sum(s ** 2)
                obj_diff = obj - obj_old
                accepted = obj_diff <= - 0.5 * self.sigma * L * ss

                # Sufficient decrease condition
                if accepted:
                    if self.verbose >= 2:
                        print "Accepted at", tt + 1
                        print "obj_diff =", obj_diff
                    break
                else:
                    L *= self.eta
            # end for line search

            L *= self.L_factor
            L = min(self.Lmax, max(self.Lmin, L))

            # Callback might need self.coef_.
            self.coef_ = coef
            if self.callback is not None:
                ret = self.callback(self)
                if ret is not None:
                    break

        return self
