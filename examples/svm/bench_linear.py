import sys
import time

import numpy as np

from scipy.stats import ttest_ind, mannwhitneyu, wilcoxon, ttest_1samp

from sklearn.base import clone
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedShuffleSplit

from lightning.datasets import load_dataset
from lightning.primal_cd import PrimalLinearSVC

from sklearn.externals.joblib import Memory
from lightning.datasets import get_data_home

memory = Memory(cachedir=get_data_home(), verbose=0, compress=6)

cache_mb = 1000
verbose = 0

@memory.cache
def fit(clf, param_dict, X_tr, y_tr, X_te, y_te, n_folds=5, n_tests=10):
    sys.stderr.write("Choose best parameter combination\n")
    cv = StratifiedShuffleSplit(y_tr, n_iterations=n_folds,
                                test_size=1.0/n_folds)
    gs = GridSearchCV(clf, param_dict, cv=cv)
    start = time.time()
    gs.fit(X_tr, y_tr)
    cv_time = time.time() - start
    best_params = gs.best_estimator_.get_params()

    sys.stderr.write("Train and test several times\n")
    times = []
    scores = []

    for i in xrange(n_tests):
        clf = clone(clf)
        clf.set_params(**best_params)
        start = time.time()
        clf.fit(X_tr, y_tr)
        times.append(time.time() - start)
        scores.append(clf.score(X_te, y_te))

    return scores, times, best_params, cv_time

try:
    dataset = sys.argv[1]
except:
    dataset = "usps0"

X_train, y_train, X_test, y_test = load_dataset(dataset)

if X_test is None:
    cv = StratifiedShuffleSplit(y_train, n_iterations=1, test_size=0.25,
                                random_state=0)
    tr, te = iter(cv).next()
    X_test, y_test = X_train[te], y_train[te]
    X_train, y_train = X_train[tr], y_train[tr]

param_dict = {
  'C' : [0.5, 1.0, 1.5, 2.0]
}

clfs = (
    ("L2L L1R Primal (Linear)", PrimalLinearSVC(penalty="l2")),
)

clf_names = zip(*clfs)[0]

results = {}
for name, clf in clfs:
    scores, times, best_params, cv_time = fit(clone(clf), param_dict,
                                              X_train, y_train,
                                              X_test, y_test)
    print name
    print "Average acc:", np.mean(scores)
    print "Std deviation:", np.std(scores)
    print "Average time:", np.mean(times)
    print "CV time", cv_time
    print best_params
