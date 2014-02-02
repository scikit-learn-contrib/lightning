.. -*- mode: rst -*-

lightning
==========

Large-scale linear classification and regression in Python/Cython.

lightning follows the `scikit-learn <http://scikit-learn.org>`_ API conventions.

Dependencies
============

The required dependencies to build the software are Python >= 2.7, setuptools,
Numpy >= 1.3, SciPy >= 0.7, scikit-learn >= 0.14 and a working C/C++ compiler.

To run the tests you will also need nose >= 0.10.

Install
=======

To install for all users on Unix/Linux::

  python setup.py build
  sudo python setup.py install

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
