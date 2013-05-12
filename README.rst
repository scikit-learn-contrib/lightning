.. -*- mode: rst -*-

lightning
==========

Large-scale sparse linear classification and regression in Python/Cython.

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

Citation
========

If you used lightning in your research, please consider citing the following paper:

| Block Coordinate Descent Algorithms for Large-scale Sparse Multiclass Classiﬁcation. [`BibTeX <http://www.mblondel.org/publications/bib/mblondel-mlj2013.txt>`_]
| Mathieu Blondel, Kazuhiro Seki, and Kuniaki Uehara.
| Machine Learning, May 2013.

For more information, see http://www.mblondel.org/journal/2013/05/12/large-scale-sparse-multiclass-classification/
