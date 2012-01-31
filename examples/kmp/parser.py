# Author: Mathieu Blondel
# License: BSD

from optparse import OptionParser

from sklearn.utils import check_random_state

def parse_kmp():
    op = OptionParser()
    op.add_option("--seed", action="store", dest="random_state", type="int")
    op.add_option("-n", action="store", default=200, dest="n_nonzero_coefs",
                  type="int")
    op.add_option("--n_components", action="store", default=0.5,
                  dest="n_components", type="int")
    op.add_option("--metric", action="store", default="rbf", dest="metric",
                  type="str")
    op.add_option("--gamma", action="store", default=0.1, dest="gamma",
                  type="str")
    op.add_option("--degree", action="store", default=4, dest="degree",
                  type="str")
    op.add_option("--coef0", action="store", default=1, dest="coef0",
                  type="str")
    op.add_option("--epsilon", action="store", default=0.001, dest="epsilon",
                  type="float")
    op.add_option("--n_validate", action="store", default=5, dest="n_validate",
                  type="int")
    op.add_option("--n_refit", action="store", default=5, dest="n_refit",
                  type="int")
    op.add_option("--alpha", action="store", default=0.1, dest="alpha",
                  type="float")
    op.add_option("--scale", action="store_true", default=False, dest="scale")


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
