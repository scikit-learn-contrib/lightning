
"""
========================================
L1 regression: regularization paths
========================================

Shows that the regularization paths obtained by coordinate descent (penalized)
and Frank-Wolfe (constrained) are equivalent.
"""
print __doc__
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from lightning.regression import CDRegressor
from lightning.regression import FWRegressor

diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

X_tr, X_te, y_tr, y_te = train_test_split(X, y, train_size=0.75, random_state=0)

plt.figure()

betas = np.logspace(-2, 5, 50)
alphas = np.logspace(-4, 4, 50)

fw_n_nz = []
fw_error = []
cd_n_nz = []
cd_error = []

for beta in betas:
    reg = FWRegressor(beta=beta, max_iter=1000, tol=1e-3, verbose=0)
    reg.fit(X_tr, y_tr)
    y_pred = reg.predict(X_te)
    fw_n_nz.append(np.sum(reg.coef_ != 0))
    fw_error.append(np.sqrt(np.mean((y_te - y_pred) ** 2)))

for alpha in alphas:
    reg = CDRegressor(alpha=alpha, penalty="l1", max_iter=1000, tol=1e-3,
                      verbose=0)
    reg.fit(X_tr, y_tr)
    y_pred = reg.predict(X_te)
    cd_n_nz.append(np.sum(reg.coef_ != 0))
    cd_error.append(np.sqrt(np.mean((y_te - y_pred) ** 2)))

plt.plot(fw_n_nz, fw_error, label="Frank-Wolfe", linewidth=3)
plt.plot(cd_n_nz, cd_error, label="Coordinate Descent", linewidth=3, linestyle="--")

plt.xlabel("Number of non-zero coefficients")
plt.ylabel("RMSE")
plt.xlim((0, X_tr.shape[1]))
#plt.ylim((160, 170))
plt.legend()

plt.show()
