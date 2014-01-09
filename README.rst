.. -*- mode: rst -*-

lightning
==========

Large-scale linear classification and regression in Python/Cython.

lightning follows the `scikit-learn <http://scikit-learn.org>`_ API conventions.

Dependencies
============

The required dependencies to build the software are Python >= 2.6,
setuptools, Numpy >= 1.3, SciPy >= 0.7, scikit-learn (git version), Cython >= 0.15.1
and a working C/C++ compiler.

To run the tests you will also need nose >= 0.10.

Install
=======

First run::

  make cython

Then to install in your home directory, use::

  python setup.py install --home

To install for all users on Unix/Linux::

  python setup.py build
  sudo python setup.py install


Documentation
=============

Classification:

* `CDClassifier <http://mblondel.org/livedoc/g/mblondel/lightning/master/lightning/primal_cd.py>`_ Learning linear classifiers by coordinate descent in the primal. Supports different losses and penalties.
* `SGDClassifier <http://mblondel.org/livedoc/g/mblondel/lightning/master/lightning/sgd.py>`_ Learning linear classifiers by stochastic (sub)gradient descent. Supports different losses and penalties.
* `LinearSVC <http://mblondel.org/livedoc/g/mblondel/lightning/master/lightning/dual_cd.py>`_ Learning linear SVM by coordinate descent in the dual. Supports optimizing for accuracy or AUC.
* `KernelSVC <http://mblondel.org/livedoc/g/mblondel/lightning/master/lightning/primal_newton.py>`_ Learning kernel SVM by Newton's method in the primal. Nice if kernel matrix fits is memory.

Regression:

* `CDRegressor <http://mblondel.org/livedoc/g/mblondel/lightning/master/lightning/primal_cd.py#class-CDRegressor>`_ Learning linear regressors by coordinate descent in the primal. Supports different losses and penalties.
* `SGDRegressor <http://mblondel.org/livedoc/g/mblondel/lightning/master/lightning/sgd.py#class-SGDRegressor>`_ Learning linear regressors by stochastic (sub)gradient descent. Supports different losses and penalties.
* `LinearSVR <http://mblondel.org/livedoc/g/mblondel/lightning/master/lightning/dual_cd.py>`_ Learning linear SVR by coordinate descent in the dual.


Citation
========

If you used lightning in your research, please consider citing the following paper:

| Block Coordinate Descent Algorithms for Large-scale Sparse Multiclass Classiﬁcation. [`BibTeX <http://www.mblondel.org/publications/bib/mblondel-mlj2013.txt>`_]
| Mathieu Blondel, Kazuhiro Seki, and Kuniaki Uehara.
| Machine Learning, May 2013.

For more information, see http://www.mblondel.org/code/mlj2013/

Author
=======

Mathieu Blondel, 2012-present
