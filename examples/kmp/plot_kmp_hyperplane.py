# Author: Mathieu Blondel
# License: BSD
"""
===================================
Kernel Matching Pursuit hyperplane
===================================

"""
print __doc__

import sys

import numpy as np
import pylab as pl
import matplotlib

from sklearn.linear_model import Ridge
from sklearn.utils import check_random_state

from lightning.kmp import KMPClassifier


random_state = check_random_state(0)


def gen_non_lin_separable_data():
    mean1 = [-1, 2]
    mean2 = [1, -1]
    mean3 = [4, -4]
    mean4 = [-4, 4]
    cov = [[1.2,0.8], [0.8, 1.2]]
    X1 = random_state.multivariate_normal(mean1, cov, 50)
    X1 = np.vstack((X1, random_state.multivariate_normal(mean3, cov, 50)))
    y1 = np.ones(len(X1))
    X2 = random_state.multivariate_normal(mean2, cov, 50)
    X2 = np.vstack((X2, random_state.multivariate_normal(mean4, cov, 50)))
    y2 = np.ones(len(X2)) * -1
    return X1, y1, X2, y2


def split_train(X1, y1, X2, y2):
    X1_train = X1[:90]
    y1_train = y1[:90]
    X2_train = X2[:90]
    y2_train = y2[:90]
    X_train = np.vstack((X1_train, X2_train))
    y_train = np.hstack((y1_train, y2_train))
    return X_train, y_train


def split_test(X1, y1, X2, y2):
    X1_test = X1[90:]
    y1_test = y1[90:]
    X2_test = X2[90:]
    y2_test = y2[90:]
    X_test = np.vstack((X1_test, X2_test))
    y_test = np.hstack((y1_test, y2_test))
    return X_test, y_test


def plot_contour(X_train, clf, color=True, surface=True):
    if color:
        pos_color = "ro"
        neg_color = "bo"
    else:
        pos_color = "wo"
        neg_color = "ko"


    pl.plot(X_train[y_train == 1, 0], X_train[y_train == 1, 1], pos_color)
    pl.plot(X_train[y_train == -1, 0], X_train[y_train == -1, 1], neg_color)
    pl.scatter(clf.components_[:,0], clf.components_[:,1],
               s=80, edgecolors="k", facecolors="none")

    X1, X2 = np.meshgrid(np.linspace(-8,8,100), np.linspace(-8,8,100))
    X = np.array([[x1, x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
    Z = clf.decision_function(X).reshape(X1.shape)

    if surface:
        # plot surface
        pl.contourf(X1, X2, Z, 10,
                    cmap=matplotlib.cm.bone, origin='lower',
                    alpha=0.85)
        pl.contour(X1, X2, Z, [0.0],
                   colors='k',
                   linestyles=['solid'])
    else:
        # plot hyperplane
        levels = [-1.0, 0.0, 1.0]
        linestyles = ['dashed', 'solid', 'dashed']
        colors = ['grey', 'black', 'grey']
        pl.contour(X1, X2, Z, levels, colors=colors, linestyles=linestyles,
                   origin="lower")

    pl.axis("tight")
    pl.show()


try:
    color = int(sys.argv[1])
except:
    color = True

try:
    surface = int(sys.argv[2])
except:
    surface = False


X1, y1, X2, y2 = gen_non_lin_separable_data()
X_train, y_train = split_train(X1, y1, X2, y2)
X_test, y_test = split_test(X1, y1, X2, y2)

clf = KMPClassifier(n_nonzero_coefs=0.3,
                    n_components=1.0,
                    metric="rbf",
                    gamma=1.0,
                    n_refit=1,
                    estimator=Ridge(alpha=0.01),
                    random_state=random_state)
clf.fit(X_train, y_train)

y_predict = clf.predict(X_test)
correct = np.sum(y_predict == y_test)
print "%d out of %d predictions correct" % (correct, len(y_predict))

plot_contour(X_train, clf, color, surface)

