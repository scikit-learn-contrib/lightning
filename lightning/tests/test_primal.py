import numpy as np
import scipy.sparse as sp

from numpy.testing import assert_array_equal, assert_array_almost_equal, \
                          assert_almost_equal
from nose.tools import assert_raises, assert_true

from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
from sklearn.datasets.samples_generator import make_classification
from sklearn.utils import check_random_state

from lightning.primal import PrimalClassifier

iris = load_iris()
random_state = check_random_state(0)
perm = random_state.permutation(iris.target.size)
iris_dense = iris.data[perm]
iris_target = iris.target[perm]
iris_sparse = sp.csr_matrix(iris_dense)

gen_dense, gen_target = make_classification(n_samples=200, n_features=100,
                                            random_state=0)
gen_sparse = sp.csr_matrix(gen_dense)

def test_primal_fit_binary():
    for X in (gen_dense, gen_sparse):
        clf = PrimalClassifier(LinearSVC())
        y_pred = clf.fit(X, gen_target).predict(X)
        assert_true(np.mean(y_pred == gen_target) >= 0.95)


def test_primal_fit_multiclass():
    for X in (iris_dense, iris_sparse):
        clf = PrimalClassifier(LinearSVC())
        y_pred = clf.fit(X, iris_target).predict(X)
        assert_true(np.mean(y_pred == iris_target))
