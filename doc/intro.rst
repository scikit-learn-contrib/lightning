Introduction
=============

lightning is composed of three modules: classification, regression and ranking.
Several solvers are available from these modules.

Primal coordinate descent
-------------------------

:class:`CDClassifier`, :class:`CDRegressor`

- Main idea: update a single coordinate at a time (closed-form update when possible, coordinate-wise gradient descent otherwise)
- Non-smooth losses: No
- Penalties: L2, L1, L1/L2
- Learning rate: no
- Multiclass: one-vs-rest, multiclass logistic, multiclass squared hinge

Dual coordinate ascent
-----------------------

:class:`LinearSVC`, :class:`LinearSVR` (L2-regularization, supports shrinking)

:class:`SDCAClassifier`, :class:`SDCARegressor` (Elastic-net, supports many losses)

- Main idea: update a single dual coordinate at a time (closed-form solution available for many loss functions)
- Non-smooth losses: Yes
- Penalties: L2, Elastic-net
- Learning rate: no
- Multiclass: one-vs-rest

FISTA
-----

:class:`FistaClassifier`

- Main idea: accelerated proximal gradient method (uses full gradients)
- Non-smooth losses: No
- Penalties: L1, L1/L2, Trace/Nuclear
- Learning rate: no
- Multiclass: one-vs-rest, multiclass logistic, multiclass squared hinge

Stochastic gradient method (SGD)
--------------------------------

:class:`SGDClassifier`, :class:`SGDRegressor`

- Main idea: replace full gradient with stochastic estimate obtained from a single sample
- Non-smooth losses: Yes
- Penalties: L2, L1, L1/L2
- Learning rate: yes (very sensitive)
- Multiclass: one-vs-rest, multiclass logistic, multiclass squared hinge

AdaGrad
-------

:class:`AdaGradClassifier`, :class:`AdaGradRegressor`

- Main idea: use per-feature learning rates (frequently occurring features in the gradients get small learning rates and infrequent features get higher ones)
- Non-smooth losses: Yes
- Penalties: L2, Elastic-net
- Learning rate: yes (not very sensitive)
- Multiclass: one-vs-rest

Stochastic averaged gradient (SAG)
---------------------------------

:class:`SAGClassifier`, :class:`SAGRegressor`

- Main idea: instead of using the full gradient (average of sample-wise gradients), compute gradient for a randomly selected sample and use out-dated gradients for other samples
- Non-smooth losses: No
- Penalties: L2
- Learning rate: yes (not very sensitive)
- Multiclass: one-vs-rest

Stochastic variance-reduced gradient (SVRG)
-------------------------------------------

:class:`SVRGClassifier`, :class:`SVRGRegressor`

- Main idea: compute full gradient periodically and use it to center the gradient estimate (this can be shown to reduce the variance)
- Non-smooth losses: No
- Penalties: L2
- Learning rate: yes (not very sensitive)
- Multiclass: one-vs-rest

PRank
------

:class:`PRank`, :class:`KernelPRank`

- Main idea: Perceptron-like algorithm for ordinal regression
- Penalties: L2
- Learning rate: no


.. toctree::
    :hidden:

    auto_examples/index
    references.rst
