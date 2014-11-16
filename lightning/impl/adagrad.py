import numpy as np

from sklearn.base import ClassifierMixin
from sklearn.utils import check_random_state
from sklearn.preprocessing import LabelBinarizer

from .base import BaseClassifier
from .dataset_fast import get_dataset
from .adagrad_fast import _adagrad_fit


from .sgd_fast import ModifiedHuber
from .sgd_fast import Hinge
from .sgd_fast import SquaredHinge
from .sgd_fast import Log
from .sgd_fast import SquaredLoss


class AdaGradClassifier(BaseClassifier, ClassifierMixin):
    """
    Solves the following objective by AdaGrad:

        minimize_w  1 / n_samples * \sum_i loss(w^T x_i, y_i)
                    + alpha * l1_ratio * ||w||_1
                    + alpha * (1 - l1_ratio) * 0.5 * ||w||^2_2
    """

    def __init__(self, eta=1.0, alpha=1.0, l1_ratio=0, loss="hinge", n_iter=10,
                 random_state=None):
        self.eta = eta
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.loss = loss
        self.n_iter = n_iter
        self.random_state = random_state

    def _get_loss(self):
        losses = {
            "modified_huber": ModifiedHuber(),
            "hinge": Hinge(1.0),
            "squared_hinge": SquaredHinge(1.0),
            "perceptron": Hinge(0.0),
            "log": Log(),
            "squared": SquaredLoss(),
        }
        return losses[self.loss]

    def fit(self, X, y):
        n_samples, n_features = X.shape
        rng = check_random_state(self.random_state)

        self.label_binarizer_ = LabelBinarizer(neg_label=-1, pos_label=1)
        Y = np.asfortranarray(self.label_binarizer_.fit_transform(y),
                              dtype=np.float64)
        n_vectors = Y.shape[1]

        ds = get_dataset(X, order="c")
        y = np.array(y, dtype=np.float64)

        self.coef_ = np.zeros((n_vectors, n_features), dtype=np.float64)
        g_sum = np.zeros((n_vectors, n_features), dtype=np.float64)
        g_norms = np.zeros((n_vectors, n_features), dtype=np.float64)

        delta = 0
        alpha1 = self.l1_ratio * self.alpha
        alpha2 = (1 - self.l1_ratio) * self.alpha
        loss = self._get_loss()

        for i in xrange(n_vectors):
            _adagrad_fit(ds, y, self.coef_[i], g_sum[i], g_norms[i], loss,
                         self.eta, delta, alpha1, alpha2, self.n_iter, rng)

        return self
