Introduction
=============

lightning is composed of three modules: classification, regression and ranking.
Several solvers are available from these modules.

Primal coordinate descent
-------------------------

:class:`CDClassifier`, :class:`CDRegressor`

Update a single coordinate at a time. Closed-form solution available for the
squared loss. Coordinate-wise (proximal) gradient descent used for other
losses. Many losses and penalties (ridge, lasso, group-lasso) supported. No
learning rate.

Dual coordinate ascent
-----------------------

:class:`LinearSVC`, :class:`LinearSVR`, :class:`LinearRidge` (L2-regularization, supports shrinking)

:class:`ProxSDCA_Classifier` (Elastic-net, supports many losses)

Update a single dual coordinate at a time. Closed-form solution available for
many loss functions. No learning rate.

FISTA
-----

:class:`FistaClassifier`

Accelerated proximal method. Uses full gradients, no learning rate. Several penalties supported.

Stochastic gradient method (SGD)
--------------------------------

:class:`SGDClassifier`, :class:`SGDRegressor`

Replace full gradient with stochastic estimate obtained from a single sample.
Several losses and penalties supported.  Very sensitive to the choice of
learning rate.

AdaGrad
-------

:class:`AdaGradClassifier`

AdaGrad uses per-feature learning rates. Frequently occurring features in the
gradients get small learning rates and infrequent features get higher ones.
Less sensitive to learning rate. Elastic-net supported.

Stochastic averaged gradient (SAG)
---------------------------------

:class:`SAGClassifier`

Instead of using the full gradient (average of sample-wise gradients), compute
gradient for a randomly selected sample and use out-dated gradients for other
samples. Less sensitive to learning rate. L2 regularization only.

Stochastic variance-reduced gradient (SVRG)
-------------------------------------------

:class:`SVRGClassifier`

Computes the full-gradient periodically. This allows to center the gradient
estimate and reduce its variance. Less sensitive to learning rate.  L2
regularization only.

PRank
------

:class:`PRank`, :class:`KernelPRank`

Simple ordinal regression method.


.. toctree::
    :hidden:

    auto_examples/index
    references.rst
