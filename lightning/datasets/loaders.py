# Author: Mathieu Blondel
# License: BSD
import os

import numpy as np

try:
    from svmlight_loader import load_svmlight_files
except ImportError:
    from sklearn.datasets import load_svmlight_files

from sklearn.datasets.base import get_data_home as _get_data_home
from sklearn.cross_validation import ShuffleSplit
from sklearn.utils import check_random_state


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

# regression

def load_abalone():
    data_home = get_data_home()
    train_file = os.path.join(data_home, "abalone_scale")
    return _todense(_load(train_file, None, "abalone"))


def load_cadata():
    data_home = get_data_home()
    train_file = os.path.join(data_home, "cadata")
    return _todense(_load(train_file, None, "cadata"))


def load_cpusmall():
    data_home = get_data_home()
    train_file = os.path.join(data_home, "cpusmall_scale")
    return _todense(_load(train_file, None, "cpusmall"))


def load_space_ga():
    data_home = get_data_home()
    train_file = os.path.join(data_home, "space_ga_scale")
    return _todense(_load(train_file, None, "space_ga"))


def load_YearPredictionMSD():
    data_home = get_data_home()
    train_file = os.path.join(data_home, "YearPredictionMSD")
    test_file = os.path.join(data_home, "YearPredictionMSD.t")
    return _todense(_load(train_file, test_file, "YearPredictionMSD"))

# binary classification

def load_adult():
    data_home = get_data_home()
    train_file = os.path.join(data_home, "adult", "adult.trn")
    test_file = os.path.join(data_home, "adult", "adult.tst")
    return _todense(_load(train_file, test_file, "adult"))


def load_banana():
    data_home = get_data_home()
    train_file = os.path.join(data_home, "banana", "banana.all.txt")
    return _todense(_load(train_file, None, "banana"))


def load_covtype():
    data_home = get_data_home()
    train_file = os.path.join(data_home, "covtype.libsvm.binary.scale")
    return _todense(_load(train_file, None, "covtype"))


def load_mnist8():
    X_train, y_train, X_test, y_test = load_mnist()
    selected = y_train == 8
    y_train[selected] = 1
    y_train[~selected] = 0
    selected = y_test == 8
    y_test[selected] = 1
    y_test[~selected] = 0
    return X_train, y_train, X_test, y_test


def load_reuters():
    data_home = get_data_home()
    train_file = os.path.join(data_home, "reuters", "money-fx.trn")
    test_file = os.path.join(data_home, "reuters", "money-fx.tst")
    return _load(train_file, test_file, "reuters")


def load_usps0():
    X_train, y_train, X_test, y_test = load_usps()
    selected = y_train == 10
    y_train[selected] = 1
    y_train[~selected] = 0
    selected = y_test == 10
    y_test[selected] = 1
    y_test[~selected] = 0
    return X_train, y_train, X_test, y_test


def load_usps0_noisy():
    X_train, y_train, X_test, y_test = load_usps0()
    n_samples = X_train.shape[0]
    indices = np.arange(n_samples)
    random_state = check_random_state(0)
    random_state.shuffle(indices)
    n = n_samples / 10
    indices = indices[:n]
    y_train[indices] = np.logical_not(y_train[indices]).astype(int)
    return X_train, y_train, X_test, y_test


def load_waveform():
    data_home = get_data_home()
    train_file = os.path.join(data_home, "waveform", "waveform.all.txt")
    return _todense(_load(train_file, None, "waveform"))

# multi-class

def load_dna():
    data_home = get_data_home()
    train_file = os.path.join(data_home, "dna.scale.tr")
    test_file = os.path.join(data_home, "dna.scale.t")
    return _todense(_load(train_file, test_file, "dna"))


def load_letter():
    data_home = get_data_home()
    train_file = os.path.join(data_home, "letter.scale.tr")
    test_file = os.path.join(data_home, "letter.scale.t")
    return _todense(_load(train_file, test_file, "letter"))


def load_mnist():
    data_home = get_data_home()
    train_file = os.path.join(data_home, "mnist.scale")
    test_file = os.path.join(data_home, "mnist.scale.t")
    return _todense(_load(train_file, test_file, "mnist"))


def load_news20():
    data_home = get_data_home()
    train_file = os.path.join(data_home, "news20.scale")
    test_file = os.path.join(data_home, "news20.t.scale")
    return _load(train_file, test_file, "news20")


def load_pendigits():
    data_home = get_data_home()
    train_file = os.path.join(data_home, "pendigits")
    test_file = os.path.join(data_home, "pendigits.t")
    return _todense(_load(train_file, test_file, "pendigits"))


def load_protein():
    data_home = get_data_home()
    train_file = os.path.join(data_home, "protein.tr")
    test_file = os.path.join(data_home, "protein.t")
    return _todense(_load(train_file, test_file, "protein"))


def load_satimage():
    data_home = get_data_home()
    train_file = os.path.join(data_home, "satimage.scale.tr")
    test_file = os.path.join(data_home, "satimage.scale.t")
    return _todense(_load(train_file, test_file, "satimage"))


def load_usps():
    data_home = get_data_home()
    train_file = os.path.join(data_home, "usps")
    test_file = os.path.join(data_home, "usps.t")
    return _todense(_load(train_file, test_file, "usps"))


def load_usps_noisy():
    X_train, y_train, X_test, y_test = load_usps()
    n_samples = X_train.shape[0]
    n = n_samples / 10
    random_state = check_random_state(0)
    indices = np.arange(n_samples)
    random_state.shuffle(indices)
    indices2 = np.arange(n_samples)
    random_state.shuffle(indices2)
    y_train[indices[:n]] = y_train[indices2[:n]]
    return X_train, y_train, X_test, y_test



LOADERS = {
            # regression
            "abalone": load_abalone,
            "cpusmall": load_cpusmall,
            "cadata": load_cadata,
            "space_ga": load_space_ga,
            "YearPredictionMSD": load_YearPredictionMSD,
            # binary classification
            "adult": load_adult,
            "banana": load_banana,
            "covtype": load_covtype,
            "mnist8": load_mnist8,
            "reuters": load_reuters,
            "usps0": load_usps0,
            "usps0_noisy": load_usps0_noisy,
            "waveform": load_waveform,
            # multi-class
            "dna": load_dna,
            "letter": load_letter,
            "news20" : load_news20,
            "mnist": load_mnist,
            "satimage": load_satimage,
            "pendigits": load_pendigits,
            "protein": load_protein,
            "usps": load_usps,
            "usps_noisy": load_usps_noisy,
}


def get_loader(dataset):
    return LOADERS[dataset]


def load_dataset(dataset):
    return get_loader(dataset)()


