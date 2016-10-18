import numpy as np
import scipy.sparse as sp
from sklearn.externals.six.moves import xrange

from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils import check_random_state
from sklearn.utils import shuffle as shuffle_func
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer

try:
    from sklearn.model_selection import ShuffleSplit
except ImportError:
    from sklearn.cross_validation import ShuffleSplit


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
        if not dense:
            X_train.sort_indices()
            X_test.sort_indices()
    else:
        X_train, y_train = X, y
        if not dense:
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


def make_classification(n_samples=100, n_features=20, n_informative=2,
                        n_redundant=2, n_repeated=0, n_classes=2,
                        n_clusters_per_class=2, weights=None, flip_y=0.01,
                        class_sep=1.0, hypercube=True, shift=0.0, scale=1.0,
                        shuffle=True, random_state=None):
    """Generate a random n-class classification problem.

    Parameters
    ----------
    n_samples : int, optional (default=100)
        The number of samples.

    n_features : int, optional (default=20)
        The total number of features. These comprise `n_informative`
        informative features, `n_redundant` redundant features, `n_repeated`
        duplicated features and `n_features-n_informative-n_redundant-
        n_repeated` useless features drawn at random.

    n_informative : int, optional (default=2)
        The number of informative features. Each class is composed of a number
        of gaussian clusters each located around the vertices of a hypercube
        in a subspace of dimension `n_informative`. For each cluster,
        informative features are drawn independently from  N(0, 1) and then
        randomly linearly combined in order to add covariance. The clusters
        are then placed on the vertices of the hypercube.

    n_redundant : int, optional (default=2)
        The number of redundant features. These features are generated as
        random linear combinations of the informative features.

    n_repeated : int, optional (default=2)
        The number of duplicated features, drawn randomly from the informative
        and the redundant features.

    n_classes : int, optional (default=2)
        The number of classes (or labels) of the classification problem.

    n_clusters_per_class : int, optional (default=2)
        The number of clusters per class.

    weights : list of floats or None (default=None)
        The proportions of samples assigned to each class. If None, then
        classes are balanced. Note that if `len(weights) == n_classes - 1`,
        then the last class weight is automatically inferred.

    flip_y : float, optional (default=0.01)
        The fraction of samples whose class are randomly exchanged.

    class_sep : float, optional (default=1.0)
        The factor multiplying the hypercube dimension.

    hypercube : boolean, optional (default=True)
        If True, the clusters are put on the vertices of a hypercube. If
        False, the clusters are put on the vertices of a random polytope.

    shift : float or None, optional (default=0.0)
        Shift all features by the specified value. If None, then features
        are shifted by a random value drawn in [-class_sep, class_sep].

    scale : float or None, optional (default=1.0)
        Multiply all features by the specified value. If None, then features
        are scaled by a random value drawn in [1, 100]. Note that scaling
        happens after shifting.

    shuffle : boolean, optional (default=True)
        Shuffle the samples and the features.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Returns
    -------
    X : array of shape [n_samples, n_features]
        The generated samples.

    y : array of shape [n_samples]
        The integer labels for class membership of each sample.

    Notes
    -----
    The algorithm is adapted from Guyon [1] and was designed to generate
    the "Madelon" dataset.

    References
    ----------
    .. [1] I. Guyon, "Design of experiments for the NIPS 2003 variable
           selection benchmark", 2003.
    """
    from itertools import product
    from sklearn.utils import shuffle as util_shuffle

    generator = check_random_state(random_state)

    # Count features, clusters and samples
    if n_informative + n_redundant + n_repeated > n_features:
        raise ValueError("Number of informative, redundant and repeated "
                         "features must sum to less than the number of total"
                         " features")
    if 2 ** n_informative < n_classes * n_clusters_per_class:
        raise ValueError("n_classes * n_clusters_per_class must"
                         " be smaller or equal 2 ** n_informative")
    if weights and len(weights) not in [n_classes, n_classes - 1]:
        raise ValueError("Weights specified but incompatible with number "
                         "of classes.")

    n_useless = n_features - n_informative - n_redundant - n_repeated
    n_clusters = n_classes * n_clusters_per_class

    if weights and len(weights) == (n_classes - 1):
        weights.append(1.0 - sum(weights))

    if weights is None:
        weights = [1.0 / n_classes] * n_classes
        weights[-1] = 1.0 - sum(weights[:-1])

    n_samples_per_cluster = []

    for k in range(n_clusters):
        n_samples_per_cluster.append(int(n_samples * weights[k % n_classes]
                                     / n_clusters_per_class))

    for i in range(n_samples - sum(n_samples_per_cluster)):
        n_samples_per_cluster[i % n_clusters] += 1

    # Intialize X and y
    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples, dtype=np.int)

    # Build the polytope
    C = np.array(list(product([-class_sep, class_sep], repeat=n_informative)))

    generator.shuffle(C)

    if not hypercube:
        C[:n_clusters] *= generator.rand(n_clusters, 1)
        C *= generator.rand(1, n_informative)

    # Loop over all clusters
    pos = 0
    pos_end = 0

    for k in range(n_clusters):
        # Number of samples in cluster k
        n_samples_k = n_samples_per_cluster[k]

        # Define the range of samples
        pos = pos_end
        pos_end = pos + n_samples_k

        # Assign labels
        y[pos:pos_end] = k % n_classes

        # Draw features at random
        X[pos:pos_end, :n_informative] = generator.randn(n_samples_k,
                                                         n_informative)

        # Multiply by a random matrix to create co-variance of the features
        A = 2 * generator.rand(n_informative, n_informative) - 1
        X[pos:pos_end, :n_informative] = np.dot(X[pos:pos_end, :n_informative],
                                                A)

        # Shift the cluster to a vertice
        X[pos:pos_end, :n_informative] += np.tile(C[k, :], (n_samples_k, 1))

    # Create redundant features
    if n_redundant > 0:
        B = 2 * generator.rand(n_informative, n_redundant) - 1
        X[:, n_informative:n_informative + n_redundant] = \
            np.dot(X[:, :n_informative], B)

    # Repeat some features
    if n_repeated > 0:
        n = n_informative + n_redundant
        indices = ((n - 1) * generator.rand(n_repeated) + 0.5).astype(np.intp)
        X[:, n:n + n_repeated] = X[:, indices]

    # Fill useless features
    X[:, n_features - n_useless:] = generator.randn(n_samples, n_useless)

    # Randomly flip labels
    if flip_y >= 0.0:
        for i in range(n_samples):
            if generator.rand() < flip_y:
                y[i] = generator.randint(n_classes)

    # Randomly shift and scale
    constant_shift = shift is not None
    constant_scale = scale is not None

    for f in range(n_features):
        if not constant_shift:
            shift = (2 * generator.rand() - 1) * class_sep

        if not constant_scale:
            scale = 1 + 100 * generator.rand()

        X[:, f] += shift
        X[:, f] *= scale

    # Randomly permute samples and features
    if shuffle:
        X, y = util_shuffle(X, y, random_state=generator)

        indices = np.arange(n_features)
        generator.shuffle(indices)
        X[:, :] = X[:, indices]

    return X, y


if __name__ == '__main__':
    X, y, w = make_nn_regression(n_samples=1000,
                                 n_features=100,
                                 n_informative=10,
                                 noise=0.05)
    print (X.data < 0).any()
    print (y < 0).any()
    print (w < 0).any()
