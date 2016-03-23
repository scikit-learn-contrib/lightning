.. -*- mode: rst -*-

.. image:: https://travis-ci.org/scikit-learn-contrib/lightning.svg?branch=master
    :target: https://travis-ci.org/scikit-learn-contrib/lightning

.. image:: https://ci.appveyor.com/api/projects/status/onn6yba9ckerlvme/branch/master?svg=true
    :target: https://ci.appveyor.com/project/fabianp/lightning-bpc6r/branch/master

lightning
==========

lightning is a library for large-scale linear classification, regression and
ranking in Python.

Highlights:

- follows the `scikit-learn <http://scikit-learn.org>`_ API conventions
- supports natively both dense and sparse data representations
- computationally demanding parts implemented in `Cython <http://cython.org>`_

Solvers supported:

- primal coordinate descent
- dual coordinate descent (SDCA, Prox-SDCA)
- SGD, AdaGrad, SAG, SAGA, SVRG
- FISTA

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
    print(clf.score(X, y))

    # Percentage of selected features
    print(clf.n_nonzero(percentage=True))

Dependencies
------------

lightning requires Python >= 2.7, setuptools, Numpy >= 1.3, SciPy >= 0.7 and
scikit-learn >= 0.15. Building from source also requires Cython and a working C/C++ compiler. To run the tests you will also need nose >= 0.10.

Installation
------------

Precompiled binaries for the stable version of this library are available for the main platforms and can be installed using pip::

    pip install sklearn-contrib-lightning

or conda::

    conda install -c https://conda.anaconda.org/scikit-learn-contrib lightning


The development version of lightning can be installed from its git repository. In this case it is assumed that you have the git version control system, a working C++ compiler, Cython and the numpy development libraries. In order to install this verion, type::

  git clone https://github.com/scikit-learn-contrib/lightning.git
  cd lightning
  python setup.py build
  sudo python setup.py install

Documentation
-------------

http://contrib.scikit-learn.org/lightning/

On Github
---------

https://github.com/scikit-learn-contrib/lightning


Authors
-------

- Mathieu Blondel, 2012-present
- Manoj Kumar, 2015-present
- Arnaud Rachez, 2016-present
- Fabian Pedregosa, 2016-present
