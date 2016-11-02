# -*- coding: utf-8 -*-
from sklearn.utils.testing import assert_array_equal, assert_equal


def check_predict_proba(clf, X):
    y_pred = clf.predict(X)
    n_samples = y_pred.shape[0]
    # normalize negative class to 0 (it is sometimes 0, sometimes -1)
    y_pred = (y_pred == 1)

    # check that predict_proba result agree with y_true
    y_proba = clf.predict_proba(X)
    assert_equal(y_proba.shape, (n_samples, 2))
    y_proba_best = (y_proba.argmax(axis=1) == 1)
    assert_array_equal(y_proba_best, y_pred)

    # check that y_proba looks like probability
    assert not (y_proba > 1).any()
    assert not (y_proba < 0).any()
    assert_array_equal(y_proba.sum(axis=1), 1.0)
