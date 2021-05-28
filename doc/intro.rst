Introduction
============

.. currentmodule:: lightning

lightning is composed of three modules: classification, regression and ranking.
Several solvers are available from each.

If you're not sure what solver to use, just go for :class:`classification.CDClassifier` /
:class:`regression.CDRegressor` or :class:`classification.SDCAClassifier` / :class:`regression.SDCARegressor`. They
are very fast and do not require any tedious tuning of a learning rate.

Primal coordinate descent
-------------------------

:class:`classification.CDClassifier`, :class:`regression.CDRegressor`

- Main idea: update a single coordinate at a time (closed-form update when possible, coordinate-wise gradient descent otherwise)
- Non-smooth losses: No
- Penalties: L2, L1, L1/L2
- Learning rate: No
- Multiclass: one-vs-rest, multiclass logistic, multiclass squared hinge

Dual coordinate ascent
----------------------

:class:`classification.LinearSVC`, :class:`regression.LinearSVR` (L2-regularization, supports shrinking)

:class:`classification.SDCAClassifier`, :class:`regression.SDCARegressor` (Elastic-net, supports many losses)

- Main idea: update a single dual coordinate at a time (closed-form solution available for many loss functions)
- Non-smooth losses: Yes
- Penalties: L2, Elastic-net
- Learning rate: No
- Multiclass: one-vs-rest

FISTA
-----

:class:`classification.FistaClassifier`, :class:`regression.FistaRegressor`

- Main idea: accelerated proximal gradient method (uses full gradients)
- Non-smooth losses: No
- Penalties: L1, L1/L2, Trace/Nuclear
- Learning rate: No
- Multiclass: one-vs-rest, multiclass logistic, multiclass squared hinge

Stochastic gradient method (SGD)
--------------------------------

:class:`classification.SGDClassifier`, :class:`regression.SGDRegressor`

- Main idea: replace full gradient with stochastic estimate obtained from a single sample
- Non-smooth losses: Yes
- Penalties: L2, L1, L1/L2
- Learning rate: Yes (very sensitive)
- Multiclass: one-vs-rest, multiclass logistic, multiclass squared hinge

AdaGrad
-------

:class:`classification.AdaGradClassifier`, :class:`regression.AdaGradRegressor`

- Main idea: use per-feature learning rates (frequently occurring features in the gradients get small learning rates and infrequent features get higher ones)
- Non-smooth losses: Yes
- Penalties: L2, Elastic-net
- Learning rate: Yes (not very sensitive)
- Multiclass: one-vs-rest

Stochastic averaged gradient (SAG and SAGA)
-------------------------------------------

:class:`classification.SAGClassifier`, :class:`classification.SAGAClassifier`, :class:`regression.SAGRegressor`, :class:`regression.SAGARegressor`

- Main idea: instead of using the full gradient (average of sample-wise gradients), compute gradient for a randomly selected sample and use out-dated gradients for other samples
- Non-smooth losses: Yes (:class:`classification.SAGAClassifier` and :class:`regression.SAGARegressor`)
- Penalties: L1, L2, Elastic-net
- Learning rate: Yes (not very sensitive)
- Multiclass: one-vs-rest

Stochastic variance-reduced gradient (SVRG)
-------------------------------------------

:class:`classification.SVRGClassifier`, :class:`regression.SVRGRegressor`

- Main idea: compute full gradient periodically and use it to center the gradient estimate (this can be shown to reduce the variance)
- Non-smooth losses: No
- Penalties: L2
- Learning rate: Yes (not very sensitive)
- Multiclass: one-vs-rest

PRank
-----

:class:`ranking.PRank`, :class:`ranking.KernelPRank`

- Main idea: Perceptron-like algorithm for ordinal regression
- Penalties: L2
- Learning rate: No


.. toctree::
    :hidden:

    auto_examples/index
    references.rst
