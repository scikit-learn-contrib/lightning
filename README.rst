.. -*- mode: rst -*-

.. image:: https://github.com/scikit-learn-contrib/lightning/actions/workflows/main.yml/badge.svg?branch=master
    :target: https://github.com/scikit-learn-contrib/lightning/actions/workflows/main.yml

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.200504.svg
    :target: https://doi.org/10.5281/zenodo.200504

lightning
==========

lightning is a library for large-scale linear classification, regression and
ranking in Python.

Highlights:

- follows the `scikit-learn <https://scikit-learn.org>`_ API conventions
- supports natively both dense and sparse data representations
- computationally demanding parts implemented in `Cython <https://cython.org>`_

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

lightning requires Python >= 3.6, setuptools, Joblib, Numpy >= 1.12, SciPy >= 0.19 and
scikit-learn >= 0.19. Building from source also requires Cython and a working C/C++ compiler. To run the tests you will also need pytest.

Installation
------------

Precompiled binaries for the stable version of lightning are available for the main platforms and can be installed using pip:

.. code-block:: sh

    pip install sklearn-contrib-lightning

or conda:

.. code-block:: sh

    conda install -c conda-forge sklearn-contrib-lightning

The development version of lightning can be installed from its git repository. In this case it is assumed that you have the git version control system, a working C++ compiler, Cython and the numpy development libraries. In order to install the development version, type:

.. code-block:: sh

  git clone https://github.com/scikit-learn-contrib/lightning.git
  cd lightning
  python setup.py install

Documentation
-------------

http://contrib.scikit-learn.org/lightning/

On GitHub
---------

https://github.com/scikit-learn-contrib/lightning

Citing
------

If you use this software, please cite it. Here is a BibTex snippet that you can use:

.. code-block::

  @misc{lightning_2016,
    author       = {Blondel, Mathieu and
                    Pedregosa, Fabian},
    title        = {{Lightning: large-scale linear classification,
                   regression and ranking in Python}},
    year         = 2016,
    doi          = {10.5281/zenodo.200504},
    url          = {https://doi.org/10.5281/zenodo.200504}
  }

Other citing formats are available in `its Zenodo entry <https://doi.org/10.5281/zenodo.200504>`_.

Authors
-------

- Mathieu Blondel
- Manoj Kumar
- Arnaud Rachez
- Fabian Pedregosa
- Nikita Titov
