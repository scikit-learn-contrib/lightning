import numpy as np

from sklearn.base import ClassifierMixin
from sklearn.utils import check_random_state
from sklearn.preprocessing import LabelBinarizer

from .base import BaseClassifier
from .dataset_fast import get_dataset
from .svrg_fast import _svrg_fit

from .sgd_fast import ModifiedHuber
from .sgd_fast import SmoothHinge
from .sgd_fast import SquaredHinge
from .sgd_fast import Log
from .sgd_fast import SquaredLoss


class SVRGClassifier(BaseClassifier, ClassifierMixin):

    def __init__(self, eta=1.0, alpha=1.0, loss="smooth_hinge", gamma=1.0,
                 max_iter=10, n_inner=1.0, tol=1e-3, verbose=0,
                 random_state=None):
        self.eta = eta
        self.alpha = alpha
        self.loss = loss
        self.gamma = gamma
        self.max_iter = max_iter
        self.n_inner = n_inner
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state

    def _get_loss(self):
        losses = {
            "modified_huber": ModifiedHuber(),
            "smooth_hinge": SmoothHinge(self.gamma),
            "squared_hinge": SquaredHinge(1.0),
            "log": Log(),
            "squared": SquaredLoss(),
        }
        return losses[self.loss]


    def fit(self, X, y):
        n_samples, n_features = X.shape

        rng = self._get_random_state()
        loss = self._get_loss()

        self.label_binarizer_ = LabelBinarizer(neg_label=-1, pos_label=1)
        Y = np.asfortranarray(self.label_binarizer_.fit_transform(y),
                              dtype=np.float64)
        n_vectors = Y.shape[1]
        n_inner = int(self.n_inner * n_samples)

        ds = get_dataset(X, order="c")

        self.coef_ = np.zeros((n_vectors, n_features), dtype=np.float64)
        full_grad = np.zeros_like(self.coef_)
        grad = np.zeros((n_vectors, n_samples), dtype=np.float64)

        for i in xrange(n_vectors):
            y = Y[:, i]

            _svrg_fit(self, ds, y, self.coef_[i], full_grad[i], grad[i],
                      self.eta, self.alpha, loss, self.max_iter, n_inner,
                      self.tol, self.verbose, rng)

        return self
