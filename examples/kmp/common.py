# Author: Mathieu Blondel
# License: BSD

from optparse import OptionParser

import numpy as np

from sklearn.utils import check_random_state


def parse_kmp(n_nonzero_coefs=200,
              n_components=0.5,
              metric="rbf",
              gamma=0.1,
              degree=4,
              coef0=1.0,
              epsilon=0.0,
              n_validate=5,
              n_refit=5,
              alpha=0.1,
              scale=False,
              scale_y=False,
              check_duplicates=False):
    op = OptionParser()
    op.add_option("--seed", action="store", dest="random_state", type="int")
    op.add_option("-n", action="store", default=n_nonzero_coefs,
                  dest="n_nonzero_coefs", type="float")
    op.add_option("--n_components", action="store", default=n_components,
                  dest="n_components", type="float")
    op.add_option("--metric", action="store", default=metric, dest="metric",
                  type="str")
    op.add_option("--gamma", action="store", default=gamma, dest="gamma",
                  type="float")
    op.add_option("--degree", action="store", default=degree, dest="degree",
                  type="int")
    op.add_option("--coef0", action="store", default=coef0, dest="coef0",
                  type="float")
    op.add_option("--epsilon", action="store", default=epsilon, dest="epsilon",
                  type="float")
    op.add_option("--n_validate", action="store", default=n_validate,
                  dest="n_validate", type="int")
    op.add_option("--n_refit", action="store", default=n_refit, dest="n_refit",
                  type="int")
    op.add_option("--alpha", action="store", default=alpha, dest="alpha",
                  type="float")
    op.add_option("--scale", action="store_true", default=scale, dest="scale")
    op.add_option("--scale_y", action="store_true", default=scale_y,
                  dest="scale_y")
    op.add_option("--check_duplicates", action="store_true",
                  default=check_duplicates, dest="check_duplicates")
    op.add_option("--regression", action="store_true", default=scale,
                  dest="regression")


    (opts, args) = op.parse_args()
    rs = check_random_state(opts.random_state)
    if opts.random_state is None:
        random_state = rs.randint(np.iinfo(np.int).max)
    else:
        random_state = opts.random_state

    try:
        dataset = args[0]
    except:
        dataset = "usps"

    return dataset, opts, random_state
