import numpy as np
import pylab as pl

from sklearn.datasets import make_regression
from sklearn.utils import check_random_state
from sklearn.linear_model import Ridge

from lightning.dual_cd import LinearSVR
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_random_state


X_train, y_train = make_regression(n_samples=15, n_features=1, n_informative=1,
                       random_state=0)

rs = check_random_state(0)
y_train += rs.normal(np.std(y_train), size=X_train.shape[0])
y_train[5] *= 5

X_test = np.linspace(-5, 5, 100).reshape(-1, 1)

pl.figure()
pl.scatter(X_train.ravel(), y_train)

reg = Ridge(alpha=1e-1)
reg.fit(X_train, y_train)
pl.plot(X_test.ravel(), reg.predict(X_test), label="Ridge")

# LinearSVR is equivalent to absolute-loss regression (robust regression)
# when epsilon=0.
reg = LinearSVR(C=10, epsilon=0, fit_intercept=True, random_state=0)
reg.fit(X_train, y_train)
pl.plot(X_test.ravel(), reg.predict(X_test), label="Robust")

pl.legend(loc="upper left")

pl.show()
