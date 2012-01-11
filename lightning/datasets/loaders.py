import os

try:
    from svmlight_loader import load_svmlight_files
except ImportError:
    from sklearn.datasets import load_svmlight_files

from sklearn.datasets.base import get_data_home as _get_data_home

def get_data_home():
    return _get_data_home().replace("scikit_learn", "lightning")

def _load(train_file, test_file, name):
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        raise IOError("Dataset missing! " +
                      "Run 'make download-%s' at the project root." % name)

    return load_svmlight_files((train_file, test_file))

def load_news20():
    data_home = get_data_home()
    train_file = os.path.join(data_home, "news20.scale")
    test_file = os.path.join(data_home, "news20.t.scale")
    return _load(train_file, test_file, "news20")

def load_usps():
    data_home = get_data_home()
    train_file = os.path.join(data_home, "usps")
    test_file = os.path.join(data_home, "usps.t")
    return _load(train_file, test_file, "usps")

def load_mnist():
    data_home = get_data_home()
    train_file = os.path.join(data_home, "mnist.scale")
    test_file = os.path.join(data_home, "mnist.scale.t")
    return _load(train_file, test_file, "mnist")
