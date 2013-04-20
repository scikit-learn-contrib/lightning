import scipy.sparse as sp

from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_greater

from sklearn.datasets.samples_generator import make_classification
from sklearn.datasets import load_digits

from lightning.sparsa import SparsaClassifier

bin_dense, bin_target = make_classification(n_samples=200, n_features=100,
                                            n_informative=5,
                                            n_classes=2, random_state=0)

mult_dense, mult_target = make_classification(n_samples=300, n_features=100,
                                              n_informative=5,
                                              n_classes=3, random_state=0)
bin_csr = sp.csr_matrix(bin_dense)
mult_csr = sp.csr_matrix(mult_dense)
digit = load_digits(2)


def test_sparsa_multiclass_l1l2():
    for data in (mult_dense, mult_csr):
        clf = SparsaClassifier(max_iter=500, penalty="l1/l2", multiclass=True)
        clf.fit(data, mult_target)
        assert_almost_equal(clf.score(data, mult_target), 0.97)


def test_sparsa_multiclass_l1():
    for data in (mult_dense, mult_csr):
        clf = SparsaClassifier(max_iter=500, penalty="l1", multiclass=True)
        clf.fit(data, mult_target)
        assert_almost_equal(clf.score(data, mult_target), 0.96)


def test_sparsa_multitask_l1l2():
    for data in (mult_dense, mult_csr):
        clf = SparsaClassifier(max_iter=500, penalty="l1/l2", multiclass=False)
        clf.fit(data, mult_target)
        assert_almost_equal(clf.score(data, mult_target), 0.88, 2)


def test_sparsa_multitask_l1():
    for data in (mult_dense, mult_csr):
        clf = SparsaClassifier(max_iter=500, penalty="l1", multiclass=False)
        clf.fit(data, mult_target)
        assert_almost_equal(clf.score(data, mult_target), 0.87, 2)


def test_sparsa_bin_l1():
    for data in (bin_dense, bin_csr):
        clf = SparsaClassifier(max_iter=500, penalty="l1")
        clf.fit(data, bin_target)
        assert_greater(clf.score(data, bin_target), 0.95)


