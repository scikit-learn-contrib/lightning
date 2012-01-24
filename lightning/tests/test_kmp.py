import numpy as np
import scipy.sparse as sp

from numpy.testing import assert_array_equal, assert_array_almost_equal, \
                          assert_almost_equal
from nose.tools import assert_raises, assert_true, assert_equal

from sklearn.datasets.samples_generator import make_classification
from sklearn.linear_model import Ridge

from lightning.kmp import KernelMatchingPursuit

bin_dense, bin_target = make_classification(n_samples=200, n_features=100,
                                            n_informative=5,
                                            n_classes=2, random_state=0)
bin_sparse = sp.csr_matrix(bin_dense)

mult_dense, mult_target = make_classification(n_samples=300, n_features=100,
                                              n_informative=5,
                                              n_classes=3, random_state=0)
mult_sparse = sp.csr_matrix(mult_dense)


def test_kmp_fit_binary():
    for metric, acc in (("rbf", 0.725),
                        ("linear", 0.825),
                        ("poly", 0.73)):
        kmp = KernelMatchingPursuit(n_nonzero_coefs=0.4,
                                    dictionary_size=0.5,
                                    metric=metric,
                                    random_state=0)
        kmp.fit(bin_dense, bin_target)
        assert_equal(kmp.dictionary_.shape[1], bin_dense.shape[0] / 2)
        y_pred = kmp.predict(bin_dense)
        assert_almost_equal(np.mean(bin_target == y_pred), acc)


def test_kmp_fit_binary_backfitting():
    for metric, acc in (("rbf", 0.725),
                        ("linear", 0.755),
                        ("poly", 0.725)):
        kmp = KernelMatchingPursuit(n_nonzero_coefs=0.5,
                                    dictionary_size=0.5,
                                    refit="backfitting",
                                    metric=metric,
                                    random_state=0)
        kmp.fit(bin_dense, bin_target)
        assert_equal(kmp.dictionary_.shape[1], bin_dense.shape[0] / 2)
        y_pred = kmp.predict(bin_dense)
        assert_almost_equal(np.mean(bin_target == y_pred), acc)


def test_kmp_fit_binary_backfitting():
    for metric, acc in (("rbf", 0.5),
                        ("linear", 0.755),
                        ("poly", 0.725)):
        kmp = KernelMatchingPursuit(n_nonzero_coefs=0.5,
                                    dictionary_size=0.5,
                                    refit="backfitting",
                                    estimator=Ridge(alpha=1.0),
                                    metric=metric,
                                    random_state=0)
        kmp.fit(bin_dense, bin_target)
        assert_equal(kmp.dictionary_.shape[1], bin_dense.shape[0] / 2)
        y_pred = kmp.predict(bin_dense)
        assert_almost_equal(np.mean(bin_target == y_pred), acc)


def test_kmp_fit_multiclass():
    for metric, acc in (("rbf", 0.79),
                        ("linear", 0.803),
                        ("poly", 0.833)):
        kmp = KernelMatchingPursuit(n_nonzero_coefs=0.4,
                                    dictionary_size=0.5,
                                    refit="backfitting",
                                    n_refit=10,
                                    metric=metric,
                                    random_state=0)
        kmp.fit(mult_dense, mult_target)
        y_pred = kmp.predict(mult_dense)
        assert_almost_equal(np.mean(mult_target == y_pred), acc, decimal=2)


def test_kmp_fit_multiclass_check_duplicates():
    for metric, acc in (("rbf", 0.80),
                        ("linear", 0.803),
                        ("poly", 0.816)):
        kmp = KernelMatchingPursuit(n_nonzero_coefs=0.4,
                                    dictionary_size=0.5,
                                    refit="backfitting",
                                    n_refit=10,
                                    check_duplicates=True,
                                    metric=metric,
                                    random_state=0)
        kmp.fit(mult_dense, mult_target)
        y_pred = kmp.predict(mult_dense)
        assert_almost_equal(np.mean(mult_target == y_pred), acc, decimal=2)

def test_kmp_squared_loss():
        kmp = KernelMatchingPursuit(n_nonzero_coefs=0.5,
                                    dictionary_size=0.5,
                                    refit="backfitting",
                                    n_refit=5,
                                    estimator=Ridge(alpha=1.0),
                                    metric="linear",
                                    random_state=0)
        kmp.fit(bin_dense, bin_target)
        y_pred = kmp.decision_function(bin_dense)

        kmp.loss = "squared"
        kmp.fit(bin_dense, bin_target)
        y_pred2 = kmp.decision_function(bin_dense)

        assert_array_almost_equal(y_pred, y_pred2)

