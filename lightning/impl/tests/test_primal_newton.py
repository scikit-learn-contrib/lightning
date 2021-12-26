import numpy as np

from lightning.impl.primal_newton import KernelSVC


def test_kernel_svc(bin_dense_train_data):
    bin_dense, bin_target = bin_dense_train_data
    clf = KernelSVC(kernel="rbf", gamma=0.1, random_state=0, verbose=0)
    clf.fit(bin_dense, bin_target)
    np.testing.assert_almost_equal(clf.score(bin_dense, bin_target), 1.0)
    assert list(clf.classes_) == [0, 1]
