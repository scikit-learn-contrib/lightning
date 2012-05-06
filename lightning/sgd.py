# Author: Mathieu Blondel
# License: BSD

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_random_state

from .sgd_fast import _linear_sgd

from .sgd_fast import ModifiedHuber
from .sgd_fast import Hinge
from .sgd_fast import Log
from .sgd_fast import SparseLog
from .sgd_fast import SquaredLoss
from .sgd_fast import Huber
from .sgd_fast import EpsilonInsensitive

class SGDClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, eta0=0.03, lmbda=0.01, loss="hinge",
                 epsilon=0.01, max_iter=10, random_state=None,
                 verbose=0, n_jobs=1):
        self.eta0 = eta0
        self.lmbda = lmbda
        self.loss = loss
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.coef_ = None

    def _get_loss(self):
        losses = {
            "modified_huber" : ModifiedHuber(),
            "hinge" : Hinge(1.0),
            "perceptron" : Hinge(0.0),
            "log": Log(),
            "sparse_log" : SparseLog(),
            "squared" : SquaredLoss(),
            "huber" : Huber(self.epsilon),
            "epsilon_insensitive" : EpsilonInsensitive(self.epsilon)
        }
        return losses[self.loss]

    def fit(self, X, y):
        n_samples, n_features = X.shape
        rs = check_random_state(self.random_state)

        self.label_binarizer_ = LabelBinarizer(neg_label=-1, pos_label=1)
        Y = self.label_binarizer_.fit_transform(y)
        n_vectors = Y.shape[1]

        self.coef_ = np.zeros((n_vectors, n_features), dtype=np.float64)

        for i in xrange(n_vectors):
            _linear_sgd(self, self.coef_[i], X, Y[:, i],
                        self._get_loss(),
                        self.lmbda, self.eta0, self.max_iter,
                        rs, self.verbose)

        return self

    def decision_function(self, X):
        return np.dot(X, self.coef_.T)

    def predict(self, X):
        pred = self.decision_function(X)
        return self.label_binarizer_.inverse_transform(pred, threshold=0)


