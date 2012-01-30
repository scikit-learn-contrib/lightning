# Author: Mathieu Blondel
# License: BSD
import os

try:
    from svmlight_loader import load_svmlight_files
except ImportError:
    from sklearn.datasets import load_svmlight_files

from sklearn.datasets.base import get_data_home as _get_data_home
from sklearn.cross_validation import ShuffleSplit


def get_data_home():
    return _get_data_home().replace("scikit_learn", "lightning")


def _load(train_file, test_file, name):
    if not os.path.exists(train_file) or \
       (test_file is not None and not os.path.exists(test_file)):
        raise IOError("Dataset missing! " +
                      "Run 'make download-%s' at the project root." % name)

    if test_file:
        return load_svmlight_files((train_file, test_file))
    else:
        X, y = load_svmlight_files((train_file,))
        return X, y, None, None


def _todense(data):
    X_train, y_train, X_test, y_test = data
    X_train = X_train.toarray()
    if X_test is not None:
        X_test = X_test.toarray()
    return X_train, y_train, X_test, y_test


def load_news20():
    data_home = get_data_home()
    train_file = os.path.join(data_home, "news20.scale")
    test_file = os.path.join(data_home, "news20.t.scale")
    return _load(train_file, test_file, "news20")


def load_usps():
    data_home = get_data_home()
    train_file = os.path.join(data_home, "usps")
    test_file = os.path.join(data_home, "usps.t")
    return _todense(_load(train_file, test_file, "usps"))


def load_usps0():
    X_train, y_train, X_test, y_test = load_usps()
    selected = y_train == 10
    y_train[selected] = 1
    y_train[~selected] = 0
    selected = y_test == 10
    y_test[selected] = 1
    y_test[~selected] = 0
    return X_train, y_train, X_test, y_test


def load_mnist():
    data_home = get_data_home()
    train_file = os.path.join(data_home, "mnist.scale")
    test_file = os.path.join(data_home, "mnist.scale.t")
    return _todense(_load(train_file, test_file, "mnist"))


def load_mnist8():
    X_train, y_train, X_test, y_test = load_mnist()
    selected = y_train == 8
    y_train[selected] = 1
    y_train[~selected] = 0
    selected = y_test == 8
    y_test[selected] = 1
    y_test[~selected] = 0
    return X_train, y_train, X_test, y_test


def load_covtype():
    data_home = get_data_home()
    train_file = os.path.join(data_home, "covtype.libsvm.binary.scale")
    return _todense(_load(train_file, None, "covtype"))


def load_adult():
    data_home = get_data_home()
    train_file = os.path.join(data_home, "adult", "adult.trn")
    test_file = os.path.join(data_home, "adult", "adult.tst")
    return _todense(_load(train_file, test_file, "adult"))


def load_reuters():
    data_home = get_data_home()
    train_file = os.path.join(data_home, "reuters", "money-fx.trn")
    test_file = os.path.join(data_home, "reuters", "money-fx.tst")
    return _load(train_file, test_file, "reuters")


def load_waveform():
    data_home = get_data_home()
    train_file = os.path.join(data_home, "waveform", "waveform.all.txt")
    return _todense(_load(train_file, None, "waveform"))


def load_banana():
    data_home = get_data_home()
    train_file = os.path.join(data_home, "banana", "banana.all.txt")
    return _todense(_load(train_file, None, "banana"))


LOADERS = { "news20" : load_news20,
            "usps": load_usps,
            "usps0": load_usps0,
            "mnist": load_mnist,
            "mnist8": load_mnist8,
            "covtype": load_covtype,
            "adult": load_adult,
            "reuters": load_reuters,
            "waveform": load_waveform,
            "banana": load_banana }


def get_loader(dataset):
    return LOADERS[dataset]


def load_dataset(dataset, proportion_train=1.0, random_state=None):
    X_train, y_train, X_test, y_test = get_loader(dataset)()

    if proportion_train < 1.0 and X_test is None:
        cv = ShuffleSplit(X_train.shape[0],
                          n_iterations=1,
                          test_fraction=1.0 - proportion_train,
                          random_state=random_state)
        train, test = list(cv)[0]
        X_tr = X_train[train]
        y_tr = y_train[train]
        X_te = X_train[test]
        y_te = y_train[test]
        X_train, y_train = X_tr, y_tr
        X_test, y_test = X_te, y_te

    return X_train, y_train, X_test, y_test
