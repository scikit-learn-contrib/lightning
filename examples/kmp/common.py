# Author: Mathieu Blondel
# License: BSD

from optparse import OptionParser

import numpy as np
import scipy.sparse as sp

from sklearn.cross_validation import KFold, StratifiedKFold, ShuffleSplit

from lightning.datasets import load_dataset

def parse_kmp(n_nonzero_coefs=200,
              n_components=0.5,
              metric="rbf",
              gamma=0.1,
              degree=4,
              coef0=1.0,
              epsilon=0.0,
              n_validate=5,
              n_refit=5,
              scale=False,
              scale_y=False,
              check_duplicates=False,
              force_cv=False,
              cvtype="random",
              bars=False,
              savefig=""):
    op = OptionParser()
    op.add_option("--n_folds", action="store", default=5,
                  dest="n_folds", type="int")
    op.add_option("-n", action="store", default=n_nonzero_coefs,
                  dest="n_nonzero_coefs", type="float")
    op.add_option("--n_components", action="store", default=n_components,
                  dest="n_components", type="float")
    op.add_option("--metric", action="store", default=metric, dest="metric",
                  type="str")
    op.add_option("--gamma", action="store", default=gamma, dest="gamma",
                  type="float")
    op.add_option("--degree", action="store", default=degree, dest="degree",
                  type="int")
    op.add_option("--coef0", action="store", default=coef0, dest="coef0",
                  type="float")
    op.add_option("--epsilon", action="store", default=epsilon, dest="epsilon",
                  type="float")
    op.add_option("--n_validate", action="store", default=n_validate,
                  dest="n_validate", type="int")
    op.add_option("--n_refit", action="store", default=n_refit, dest="n_refit",
                  type="int")
    op.add_option("--scale", action="store_true", default=scale, dest="scale")
    op.add_option("--scale_y", action="store_true", default=scale_y,
                  dest="scale_y")
    op.add_option("--check_duplicates", action="store_true",
                  default=check_duplicates, dest="check_duplicates")
    op.add_option("--regression", action="store_true", default=scale,
                  dest="regression")
    op.add_option("--force_cv", action="store_true", default=force_cv,
                  dest="force_cv")
    op.add_option("--cvtype", action="store", default=cvtype, dest="cvtype",
                  type="str")
    op.add_option("--bars", action="store_true", default=bars,
                  dest="bars")
    op.add_option("--savefig", action="store", default=savefig, dest="savefig",
                  type="str")

    (opts, args) = op.parse_args()

    try:
        dataset = args[0]
    except:
        dataset = "usps"
    try:
        X_train, y_train, X_test, y_test = load_dataset(dataset)

        print "X_train", X_train.shape
        if X_test is not None: print "X_test", X_test.shape

        return X_train, y_train, X_test, y_test, opts, args
    except KeyError:
        raise ValueError("Wrong dataset name!")


def plot(pl, x, y, yerr, label, error_bar):
    if error_bar:
        pl.errorbar(x, y, yerr=yerr, label=label)
    else:
        pl.plot(x, y, label=label)


def split_data(X_train, y_train, X_test, y_test, n_folds, cvtype="random",
               force_cv=False):
    sparse = sp.issparse(X_train)

    if X_test is not None and not force_cv:
        yield X_train, y_train, X_test, y_test
    else:
        if X_test is not None:
            X_train = np.vstack((X_train, X_test))
            y_train = np.concatenate((y_train, y_test))

        if cvtype == "stratified":
            cv = StratifiedKFold(y_train, n_folds, indices=sparse)
        elif cvtype == "kfold":
            cv = KFold(X_train.shape[0], n_folds, indices=sparse)
        elif cvtype == "random":
            cv = ShuffleSplit(X_train.shape[0], n_iterations=n_folds,
                              test_fraction=0.3, random_state=0)
        else:
            raise ValueError("Unknown cvtype")

        for train, test in cv:
            yield X_train[train], y_train[train], X_train[test], y_train[test]
