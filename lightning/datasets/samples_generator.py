import numpy as np
import scipy.sparse as sp

from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils import check_random_state
from sklearn.utils import shuffle as shuffle_func
from sklearn.cross_validation import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer


def _make_nn_regression(n_samples=100, n_features=100, n_informative=10,
                        shuffle=True, random_state=None):

    generator = check_random_state(random_state)

    row = np.repeat(np.arange(n_samples), n_informative)
    col = np.zeros(n_samples * n_informative, dtype=np.int32)
    data = generator.rand(n_samples * n_informative)

    n = 0
    ind = np.arange(n_features)
    for i in xrange(n_samples):
        generator.shuffle(ind)
        col[n:n+n_informative] = ind[:n_informative]
        n += n_informative

    X = sp.coo_matrix((data, (row, col)), shape=(n_samples, n_features))
    X = X.tocsr()

    # Generate a ground truth model with only n_informative features being non
    # zeros (the other features are not correlated to y and should be ignored
    # by a sparsifying regularizers such as L1 or elastic net)
    ground_truth = np.zeros(n_features)
    v = generator.rand(n_informative)
    v += np.min(v)
    ground_truth[:n_informative] = 100 * v
    y = safe_sparse_dot(X, ground_truth)


    # Randomly permute samples and features
    if shuffle:
        X, y = shuffle_func(X, y, random_state=generator)

    return X, y, ground_truth

def make_nn_regression(n_samples=100, n_features=100, n_informative=10,
                       dense=False, noise=0.0, test_size=0,
                       normalize_x=True, normalize_y=True,
                       shuffle=True, random_state=None):

    X, y, w = _make_nn_regression(n_samples=n_samples,
                                  n_features=n_features,
                                  n_informative=n_informative,
                                  shuffle=shuffle,
                                  random_state=random_state)

    if dense:
        X = X.toarray()

    if test_size > 0:
        cv = ShuffleSplit(len(y), n_iter=1, random_state=random_state,
                          test_size=test_size, train_size=1-test_size)

        train, test = list(cv)[0]
        X_train, y_train = X[train], y[train]
        X_test, y_test = X[test], y[test]
        X_train.sort_indices()
        X_test.sort_indices()
    else:
        X_train, y_train = X, y
        X_train.sort_indices()
        X_test, y_test = None, None

    # Add noise
    if noise > 0.0:
        generator = check_random_state(random_state)
        y_train += generator.normal(scale=noise * np.std(y_train),
                                    size=y_train.shape)
        y_train = np.maximum(y_train, 0)

    if normalize_x:
        normalizer = Normalizer()
        X_train = normalizer.fit_transform(X_train)
        if X_test is not None:
            X_test = normalizer.transform(X_test)

    if normalize_y:
        scaler = MinMaxScaler()
        y_train = scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
        if y_test is not None:
            y_test = scaler.transform(y_test.reshape(-1, 1)).ravel()

    if X_test is not None:
        return X_train, y_train, X_test, y_test, w
    else:
        return X_train, y_train, w


if __name__ == '__main__':
    X, y, w = make_nn_regression(n_samples=1000,
                                 n_features=100,
                                 n_informative=10,
                                 noise=0.05)
    print (X.data < 0).any()
    print (y < 0).any()
    print (w < 0).any()
