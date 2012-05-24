import numpy as np
import scipy.sparse as sp

from numpy.testing import assert_array_equal, assert_array_almost_equal, \
                          assert_almost_equal
from nose.tools import assert_raises, assert_true, assert_equal, \
                       assert_not_equal

from sklearn.datasets.samples_generator import make_classification
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils import check_random_state
from sklearn.preprocessing import LabelBinarizer

from lightning.primal_cd import PrimalLinearSVC, PrimalSVC
from lightning.primal_cd import C_lower_bound, C_upper_bound

from lightning.kernel_fast import get_kernel, KernelCache

bin_dense, bin_target = make_classification(n_samples=200, n_features=100,
                                            n_informative=5,
                                            n_classes=2, random_state=0)
bin_sparse = sp.csr_matrix(bin_dense)

mult_dense, mult_target = make_classification(n_samples=300, n_features=100,
                                              n_informative=5,
                                              n_classes=3, random_state=0)
mult_sparse = sp.csr_matrix(mult_dense)


def test_fit_linear_binary_l1r():
    clf = PrimalLinearSVC(C=1.0, random_state=0, penalty="l1")
    clf.fit(bin_dense, bin_target)
    acc = clf.score(bin_dense, bin_target)
    assert_almost_equal(acc, 1.0)
    n_nz = np.sum(clf.coef_ != 0)

    clf = PrimalLinearSVC(C=0.1, random_state=0, penalty="l1")
    clf.fit(bin_dense, bin_target)
    acc = clf.score(bin_dense, bin_target)
    assert_almost_equal(acc, 0.97)
    n_nz2 = np.sum(clf.coef_ != 0)

    assert_true(n_nz > n_nz2)


def test_fit_rbf_binary_l1r():
    clf = PrimalSVC(C=0.5, kernel="rbf", gamma=0.1, random_state=0, penalty="l1")
    clf.fit(bin_dense, bin_target)
    acc = clf.score(bin_dense, bin_target)
    assert_almost_equal(acc, 0.845)
    n_nz = np.sum(clf.coef_ != 0)
    assert_equal(n_nz, 160)

    K = pairwise_kernels(bin_dense, metric="rbf", gamma=0.1)
    clf2 = PrimalLinearSVC(C=0.5, random_state=0, penalty="l1")
    clf2.fit(K, bin_target)
    acc = clf2.score(K, bin_target)
    assert_almost_equal(acc, 0.845)
    n_nz = np.sum(clf2.coef_ != 0)
    assert_equal(n_nz, 160)


def test_fit_rbf_binary_l1r_selection():
    for selection in ("loss", "active"):
        clf = PrimalSVC(C=0.5, kernel="rbf", gamma=0.1, random_state=0,
                        penalty="l1", selection=selection)
        clf.fit(bin_dense, bin_target)
        acc = clf.score(bin_dense, bin_target)
        assert_true(acc >= 0.74)
        n_nz = np.sum(clf.coef_ != 0)
        assert_true(n_nz <= 85)


def test_fit_rbf_multi():
    clf = PrimalSVC(penalty="l1", kernel="rbf", gamma=0.1, random_state=0)
    clf.fit(mult_dense, mult_target)
    y_pred = clf.predict(mult_dense)
    acc = np.mean(y_pred == mult_target)
    assert_almost_equal(acc, 1.0)


def test_warm_start_l1r():
    clf = PrimalLinearSVC(warm_start=True, random_state=0, penalty="l1")

    clf.C = 0.1
    clf.fit(bin_dense, bin_target)
    n_nz = np.sum(clf.coef_ != 0)

    clf.C = 0.2
    clf.fit(bin_dense, bin_target)
    n_nz2 = np.sum(clf.coef_ != 0)

    assert_true(n_nz < n_nz2)


def test_warm_start_l1r_rbf():
    clf = PrimalSVC(warm_start=True, kernel="rbf", gamma=0.1,
                    random_state=0, penalty="l1")

    clf.C = 0.5
    clf.fit(bin_dense, bin_target)
    n_nz = np.sum(clf.coef_ != 0)

    clf.C = 0.6
    clf.fit(bin_dense, bin_target)
    n_nz2 = np.sum(clf.coef_ != 0)

    assert_true(n_nz < n_nz2)


def test_early_stopping_l1r_rbf():
    clf = PrimalSVC(kernel="rbf", gamma=0.1,
                    termination="n_sv", sv_upper_bound=30,
                    random_state=0, penalty="l1")

    clf.fit(bin_dense, bin_target)
    n_nz = np.sum(clf.coef_ != 0)

    assert_equal(n_nz, 30)


def test_fit_linear_binary_l2r():
    clf = PrimalLinearSVC(C=1.0, random_state=0, penalty="l2")
    clf.fit(bin_dense, bin_target)
    acc = clf.score(bin_dense, bin_target)
    assert_almost_equal(acc, 1.0)

    K = pairwise_kernels(bin_dense, metric="rbf", gamma=0.1)
    clf2 = PrimalLinearSVC(C=0.5, random_state=0, penalty="l2")
    clf2.fit(K, bin_target)
    acc = clf2.score(K, bin_target)
    assert_almost_equal(acc, 1.0)
    n_nz = np.sum(clf2.coef_ != 0)
    assert_equal(n_nz, 200)


def test_fit_linear_binary_l2r_log():
    clf = PrimalLinearSVC(C=1.0, random_state=0, penalty="l2", loss="log",
                          max_iter=5)
    clf.fit(bin_dense, bin_target)
    acc = clf.score(bin_dense, bin_target)
    assert_almost_equal(acc, 1.0)


def test_fit_rbf_binary_l2r_log():
    clf = PrimalSVC(C=1.0, random_state=0, penalty="l2", loss="log",
                          max_iter=5, kernel="rbf")
    clf.fit(bin_dense, bin_target)
    acc = clf.score(bin_dense, bin_target)
    assert_almost_equal(acc, 1.0)


def test_fit_linear_binary_l2r_modified_huber():
    clf = PrimalLinearSVC(C=1.0, random_state=0, penalty="l2",
                          loss="modified_huber")
    clf.fit(bin_dense, bin_target)
    acc = clf.score(bin_dense, bin_target)
    assert_almost_equal(acc, 1.0)


def test_fit_rbf_binary_l2r_modified_huber():
    clf = PrimalSVC(C=1.0, random_state=0, penalty="l2",
                    kernel="rbf", loss="modified_huber")
    clf.fit(bin_dense, bin_target)
    acc = clf.score(bin_dense, bin_target)
    assert_almost_equal(acc, 1.0)


def test_fit_rbf_binary_l2r():
    clf = PrimalSVC(C=0.5, kernel="rbf", gamma=0.1, random_state=0, penalty="l2")
    clf.fit(bin_dense, bin_target)
    acc = clf.score(bin_dense, bin_target)
    assert_almost_equal(acc, 1.0)
    n_nz = np.sum(clf.coef_ != 0)
    assert_equal(n_nz, 200) # dense solution...


def test_warm_start_l2r():
    clf = PrimalLinearSVC(warm_start=True, random_state=0, penalty="l2")

    clf.C = 0.1
    clf.fit(bin_dense, bin_target)
    assert_almost_equal(clf.score(bin_dense, bin_target), 1.0)

    clf.C = 0.2
    clf.fit(bin_dense, bin_target)
    assert_almost_equal(clf.score(bin_dense, bin_target), 1.0)


def test_warm_start_l2r_rbf():
    clf = PrimalSVC(warm_start=True, kernel="rbf", gamma=0.1,
                    random_state=0, penalty="l2")

    clf.C = 0.1
    clf.fit(bin_dense, bin_target)
    assert_almost_equal(clf.score(bin_dense, bin_target), 1.0)

    clf.C = 0.2
    clf.fit(bin_dense, bin_target)
    assert_almost_equal(clf.score(bin_dense, bin_target), 1.0)


def test_debiasing():
    clf = PrimalSVC(kernel="rbf", gamma=0.1, penalty="l1l2", C=0.5, Cd=1.0,
                    max_iter=10)
    clf.fit(bin_dense, bin_target)
    assert_equal(clf.n_support_vectors(), 160)
    assert_almost_equal(clf.score(bin_dense, bin_target), 0.845)
    pred = clf.decision_function(bin_dense)

    clf = PrimalSVC(kernel="rbf", gamma=0.1, penalty="l1", C=0.5)
    clf.fit(bin_dense, bin_target)
    assert_equal(clf.n_support_vectors(), 160)
    K = pairwise_kernels(bin_dense, clf.support_vectors_, metric="rbf", gamma=0.1)
    clf = PrimalLinearSVC(max_iter=10, C=1.0)
    clf.fit(K, bin_target)
    assert_almost_equal(clf.score(K, bin_target), 0.845)
    pred2 = clf.decision_function(K)

    assert_array_almost_equal(pred, pred2)

def test_debiasing_warm_start():
    clf = PrimalSVC(kernel="rbf", gamma=0.1, penalty="l1", max_iter=10)
    clf.C = 0.5
    clf.fit(bin_dense, bin_target)
    assert_equal(clf.n_support_vectors(), 160)
    assert_almost_equal(clf.score(bin_dense, bin_target), 0.845)

    clf = PrimalSVC(kernel="rbf", gamma=0.1, penalty="l1", max_iter=10)
    clf.C = 0.500001
    clf.fit(bin_dense, bin_target)
    assert_equal(clf.n_support_vectors(), 191)
    assert_almost_equal(clf.score(bin_dense, bin_target), 0.97)

    clf = PrimalSVC(kernel="rbf", gamma=0.1, penalty="l1", max_iter=10,
                    warm_start=True)
    clf.C = 0.5
    clf.fit(bin_dense, bin_target)
    assert_equal(clf.n_support_vectors(), 160)
    assert_almost_equal(clf.score(bin_dense, bin_target), 0.845)

    clf = PrimalSVC(kernel="rbf", gamma=0.1, penalty="l1", max_iter=10,
                    warm_start=True)
    clf.C = 0.500001
    clf.fit(bin_dense, bin_target)
    assert_equal(clf.n_support_vectors(), 191)
    assert_almost_equal(clf.score(bin_dense, bin_target), 0.97)


def test_early_stopping_l2r_rbf():
    clf = PrimalSVC(kernel="rbf", gamma=0.1,
                    termination="n_sv", sv_upper_bound=30,
                    random_state=0, penalty="l2")

    clf.fit(bin_dense, bin_target)
    n_nz = np.sum(clf.coef_ != 0)

    assert_equal(n_nz, 30)


def test_empty_model():
    clf = PrimalSVC(kernel="rbf", gamma=0.1, C=0.1, penalty="l1")
    clf.fit(bin_dense, bin_target)
    assert_equal(clf.n_support_vectors(), 0)
    acc = clf.score(bin_dense, bin_target)
    assert_equal(acc, 0.5)

    clf = PrimalSVC(kernel="rbf", gamma=0.1, C=0.1, penalty="l1l2")
    clf.fit(bin_dense, bin_target)
    assert_equal(clf.n_support_vectors(), 0)
    acc = clf.score(bin_dense, bin_target)
    assert_equal(acc, 0.5)


def test_lower_bound_binary():
    Cmin = C_lower_bound(bin_dense, bin_target)
    clf = PrimalLinearSVC(C=Cmin, random_state=0, penalty="l1")
    clf.fit(bin_dense, bin_target)
    n_nz = np.sum(clf.coef_ != 0)
    assert_equal(0, n_nz)

    clf = PrimalLinearSVC(C=Cmin * 2, random_state=0, penalty="l1")
    clf.fit(bin_dense, bin_target)
    n_nz = np.sum(clf.coef_ != 0)
    assert_not_equal(0, n_nz)


def test_lower_bound_multi():
    Cmin = C_lower_bound(mult_dense, mult_target)
    assert_almost_equal(Cmin, 0.00176106681581)


def test_lower_bound_binary_rbf():
    K = pairwise_kernels(bin_dense, metric="rbf", gamma=0.1)
    Cmin = C_lower_bound(K, bin_target)
    Cmin2 = C_lower_bound(bin_dense, bin_target, kernel="rbf", gamma=0.1)
    assert_almost_equal(Cmin, Cmin2, 4)
    Cmin3 = C_lower_bound(bin_dense, bin_target, kernel="rbf", gamma=0.1,
                          search_size=60, random_state=0)
    assert_almost_equal(Cmin, Cmin3, 4)


def test_lower_bound_multi_rbf():
    K = pairwise_kernels(mult_dense, metric="rbf", gamma=0.1)
    Cmin = C_lower_bound(K, mult_target)
    Cmin2 = C_lower_bound(mult_dense, mult_target, kernel="rbf", gamma=0.1)
    Cmin3 = C_lower_bound(mult_dense, mult_target, kernel="rbf", gamma=0.1,
                          search_size=60, random_state=0)
    assert_almost_equal(Cmin, Cmin2, 4)
    assert_almost_equal(Cmin, Cmin3, 4)


def test_upper_bound_rbf():
    clf = PrimalSVC(random_state=0, penalty="l1", kernel="rbf", gamma=0.1)
    Cmin = C_lower_bound(bin_dense, bin_target, kernel="rbf", gamma=0.1)
    Cmax = C_upper_bound(bin_dense, bin_target, clf, Cmin, 5.0, 100, 10)
    clf.set_params(C=Cmax)
    clf.fit(bin_dense, bin_target)
    assert_true(clf.n_support_vectors() < 110)


def test_components():
    clf = PrimalSVC(random_state=0, penalty="l1", kernel="rbf",
                    gamma=0.1, C=0.5)
    clf.fit(bin_dense, bin_target)
    acc = clf.score(bin_dense, bin_target)

    clf = PrimalSVC(random_state=0, penalty="l2", kernel="rbf",
                    gamma=0.1, C=0.5, components=clf.support_vectors_)
    clf.fit(bin_dense, bin_target)
    assert_equal(clf.n_support_vectors(), 160)
    acc2 = clf.score(bin_dense, bin_target)
    assert_equal(acc, acc2)


def test_shared_kcache():
    clf = PrimalSVC(warm_start=True, kernel="rbf", gamma=0.1,
                    random_state=0, penalty="l1")
    kernel = get_kernel("rbf", gamma=0.1)
    kcache = KernelCache(kernel, bin_dense.shape[0], 10, 1, 0)

    clf.C = 0.5
    clf.fit(bin_dense, bin_target, kcache=kcache)
    n_nz = np.sum(clf.coef_ != 0)

    clf.C = 0.6
    clf.fit(bin_dense, bin_target, kcache=kcache)
    n_nz2 = np.sum(clf.coef_ != 0)

    assert_true(n_nz < n_nz2)


def test_fit_rbf_binary_l2r_kernelized_selection():
    clf = PrimalSVC(C=1.0, random_state=0, penalty="l2", loss="squared_hinge",
                    max_iter=1, kernel="rbf", kernel_regularizer=True,
                    selection="loss")
    clf.fit(bin_dense, bin_target)
    acc = clf.score(bin_dense, bin_target)
    assert_almost_equal(acc, 1.0)


def test_fit_rbf_binary_l2r_correctness():
    for loss in ("squared_hinge", "modified_huber", "log"):
        clf = PrimalSVC(C=1.0, random_state=0, penalty="l2", loss=loss,
                        max_iter=1, kernel="rbf", kernel_regularizer=False)
        clf.fit(bin_dense, bin_target)
        acc = clf.score(bin_dense, bin_target)
        assert_almost_equal(acc, 1.0)

        clf2 = PrimalKernelSVC(C=1.0, random_state=0, loss=loss,
                               max_iter=1, kernel="rbf",
                               kernel_regularizer=False)
        clf2.fit(bin_dense, bin_target)
        assert_array_almost_equal(clf.coef_, clf2.coef_, decimal=5)


def test_fit_rbf_binary_l2r_correctness_kernelized():
    for loss in ("squared_hinge", "modified_huber", "log"):
        clf = PrimalSVC(C=1.0, random_state=0, penalty="l2", loss=loss,
                        max_iter=1, kernel="rbf", kernel_regularizer=True)
        clf.fit(bin_dense, bin_target)
        acc = clf.score(bin_dense, bin_target)
        assert_almost_equal(acc, 1.0)

        clf2 = PrimalKernelSVC(C=1.0, random_state=0, loss=loss,
                               max_iter=1, kernel="rbf",
                               kernel_regularizer=True)
        clf2.fit(bin_dense, bin_target)
        assert_array_almost_equal(clf.coef_, clf2.coef_, decimal=5)

# Naive implementation to check correctness...
class Loss(object):

    def __init__(self, kernel_regularizer):
        self.kernel_regularizer = kernel_regularizer

class SquaredHingeLoss(Loss):

    def objective(self, K, y, coef, C):
        if self.kernel_regularizer:
            value = 0.5 * np.dot(np.dot(K, coef), coef)
        else:
            value = 0.5 * np.dot(coef, coef)
        losses = np.maximum(1 - y * np.dot(K, coef), 0) ** 2
        value += C * np.sum(losses)
        return value

    def derivative(self, K, y, coef, j, C):
        if self.kernel_regularizer:
            value = np.dot(coef, K[j])
        else:
            value = coef[j]
        losses = np.maximum(1 - y * np.dot(K, coef), 0)
        value += -2 * C * np.sum(y * K[j] * losses)
        return value

    def second_derivative(self, K, y, coef, j, C):
        if self.kernel_regularizer:
            value = K[j, j]
        else:
            value = 1
        losses = 1 - y * np.dot(K, coef)
        values = K[j] ** 2
        values[losses < 0] = 0
        value += 2 * C * np.sum(values)
        return value

class LogLoss(Loss):

    def objective(self, K, y, coef, C):
        value = 0.5 * np.dot(np.dot(K, coef), coef)
        losses = np.log(1 + np.exp(-y * np.dot(K, coef)))
        value += C * np.sum(losses)
        return value

    def derivative(self, K, y, coef, j, C):
        value = np.dot(coef, K[j])
        losses = y * np.dot(K, coef)
        losses = 1 / (1 + np.exp(-losses))
        value += C * np.sum(y * K[j] * (losses - 1))
        return value

    def second_derivative(self, K, y, coef, j, C):
        value = K[j, j]
        losses = y * np.dot(K, coef)
        losses = 1 / (1 + np.exp(-losses))
        value += C * np.sum(K[j] ** 2 * losses * (1- losses))
        return value


class ModifiedHuberLoss(Loss):

    def objective(self, K, y, coef, C):
        value = 0.5 * np.dot(np.dot(K, coef), coef)
        losses = y * np.dot(K, coef)
        cond = losses < -1
        not_cond = ~cond
        losses[cond] *= -4
        losses[not_cond] = np.maximum(1-losses[not_cond], 0) ** 2
        value += C * np.sum(losses)
        return value

    def derivative(self, K, y, coef, j, C):
        value = np.dot(coef, K[j])
        losses = y * np.dot(K, coef)
        cond = losses < -1
        not_cond = ~cond
        losses[cond] = -4
        losses[not_cond] = -2 * np.maximum(1 - losses[not_cond], 0)
        value += C * np.sum(y * K[j] * losses)
        return value

    def second_derivative(self, K, y, coef, j, C):
        value = K[j, j]
        losses = y * np.dot(K, coef)
        cond = np.logical_and(-1 <= losses, losses <= 1)
        value += 2 * C * np.sum(K[j, cond] ** 2)
        return value


class PrimalKernelSVC(PrimalSVC):

    def _get_loss(self):
        losses = {"squared_hinge" : SquaredHingeLoss(self.kernel_regularizer),
                  "log" : LogLoss(self.kernel_regularizer),
                  "modified_huber" : ModifiedHuberLoss(self.kernel_regularizer)}
        return losses[self.loss]

    def _solve_one(self, K, y, coef, j, loss):
        sigma = 0.01
        beta = 0.5
        L0 = loss.objective(K, y, coef, self.C)
        d = -loss.derivative(K, y, coef, j, self.C)
        d /= loss.second_derivative(K, y, coef, j, self.C)
        old_coef = coef[j]
        z = d

        if abs(d) <= 1e-12:
            return

        for i in xrange(100):
            coef[j] = old_coef + z
            Li = loss.objective(K, y, coef, self.C)
            if Li - L0 <= -sigma * (z ** 2):
                break
            z *= beta

    def _fit_binary(self, K, y, coef, loss, rs):
        n_samples = K.shape[0]
        indices = np.arange(n_samples)
        rs.shuffle(indices)

        for t in xrange(self.max_iter * n_samples):
            j = indices[(t-1) % n_samples]
            self._solve_one(K, y, coef, j, loss)

    def fit(self, X, y):
        n_samples = X.shape[0]
        rs = check_random_state(self.random_state)

        self.label_binarizer_ = LabelBinarizer(neg_label=-1, pos_label=1)
        Y = self.label_binarizer_.fit_transform(y)
        self.classes_ = self.label_binarizer_.classes_.astype(np.int32)
        n_vectors = Y.shape[1]

        self.coef_ = np.zeros((n_vectors, n_samples), dtype=np.float64)
        #self.errors_ = np.ones((n_vectors, n_samples), dtype=np.float64)
        self.intercept_ = np.zeros(n_vectors, dtype=np.float64)

        K = pairwise_kernels(X, metric=self.kernel, filter_params=True,
                             **self._kernel_params())
        loss = self._get_loss()

        for i in xrange(n_vectors):
            self._fit_binary(K, Y[:, i], self.coef_[i], loss, rs)

        self._post_process(X)

        return self
