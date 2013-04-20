import numpy as np

def make_ovo(X, y, class1, class2):
    classes = np.unique(y)

    if len(y) <= 2:
        return X, y

    c1 = classes[class1]
    c2 = classes[class2]
    cond = np.logical_or(y == classes[c1], y == classes[c2])
    y = y[cond]
    y[y == c1] = 0
    y[y == c2] = 1
    ind = np.arange(X.shape[0])

    return X[ind[cond]], y
