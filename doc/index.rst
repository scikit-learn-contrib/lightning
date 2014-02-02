lightning
==========

lightning is a library for large-scale linear classification and regression in
Python.

Highlights:

- follows the `scikit-learn <http://scikit-learn.org>`_ API conventions
- supports natively both dense and sparse data representations
- computationally demanding parts implemented in `Cython <http://cython.org>`_

Solvers supported:

- coordinate descent
- FISTA
- stochastic gradient descent
- SpaRSA

Loss functions supported:

- hinge
- squared hinge
- logistic
- multiclass logistic
- squared
- epsilon-insensitive
- modified-huber

Penalties supported:

- L2 (Ridge)
- L1 (Lasso)
- L1/L2 (Group Lasso)

Note: Not all solvers support all loss / penalty combinations. See the class
references for more details.

Example
-------

Example that shows how to learn a multiclass classifier with group lasso
penalty on the News20 dataset (c.f., `Blondel et al. 2013
<http://www.mblondel.org/publications/mblondel-mlj2013.pdf>`_):

.. code-block:: python

    from sklearn.datasets import fetch_20newsgroups_vectorized
    from lightning.classification import CDClassifier

    # Load News20 dataset from scikit-learn.
    bunch = fetch_20newsgroups_vectorized(subset="all")
    X = bunch.data
    y = bunch.target

    # Set classifier options.
    clf = CDClassifier(penalty="l1/l2",
                       loss="squared_hinge",
                       multiclass=True,
                       max_iter=20,
                       alpha=1e-4,
                       C=1.0 / X.shape[0],
                       tol=1e-3)

    # Train the model.
    clf.fit(X, y)

    # Accuracy
    print clf.score(X, y)

    # Percentage of selected features
    print clf.n_nonzero(percentage=True)

Dependencies
------------

lightning needs Python >= 2.7, setuptools, Numpy >= 1.3, SciPy >= 0.7,
scikit-learn >= 0.14 and a working C/C++ compiler.

To run the tests you will also need nose >= 0.10.

Installation
------------

To install lightning from pip, type::

    pip install https://github.com/mblondel/lightning/archive/master.zip

To install lightning from source, type::

  python setup.py build
  sudo python setup.py install


.. toctree::
    :hidden:

    auto_examples/index
    references.rst
