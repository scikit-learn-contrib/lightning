

.. _plot_sparse_non_linear.py:


================================
Sparse non-linear classification
================================

This examples demonstrates how to use `CDClassifier` with L1 penalty to do
sparse non-linear classification. The trick simply consists in fitting the
classifier with a kernel matrix (e.g., using an RBF kernel).

There are a few interesting differences with standard kernel SVMs:

1. the kernel matrix does not need to be positive semi-definite (hence the
expression "kernel matrix" above is an abuse of terminology)

2. the number of "support vectors" will be typically smaller thanks to L1
regularization and can be adjusted by the regularization parameter C (the
smaller C, the fewer the support vectors)

3. the "support vectors" need not be located at the margin



.. rst-class:: horizontal


    *

      .. image:: images/plot_sparse_non_linear_1.png
            :scale: 47

    *

      .. image:: images/plot_sparse_non_linear_2.png
            :scale: 47




**Python source code:** :download:`plot_sparse_non_linear.py <plot_sparse_non_linear.py>`

.. literalinclude:: plot_sparse_non_linear.py
    :lines: 21-

**Total running time of the example:**  0.18 seconds
    