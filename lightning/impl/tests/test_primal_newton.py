from sklearn.utils.testing import assert_almost_equal

from sklearn.datasets.samples_generator import make_classification

from lightning.impl.primal_newton import KernelSVC

bin_dense, bin_target = make_classification(n_samples=200, n_features=100,
                                            n_informative=5,
                                            n_classes=2, random_state=0)

def test_kernel_svc():
    clf = KernelSVC(kernel="rbf", gamma=0.1, random_state=0, verbose=0)
    clf.fit(bin_dense, bin_target)
    assert_almost_equal(clf.score(bin_dense, bin_target), 1.0)
