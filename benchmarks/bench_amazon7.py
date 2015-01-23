import sys

from sklearn.externals import joblib

from lightning.classification import SDCAClassifier

if len(sys.argv) == 1:
    print """
    Please enter the path to amazon7_uncompressed_pkl/amazon7.pkl

    Download data from
    http://www.mblondel.org/data/amazon7_uncompressed_pkl.tar.bz2
    """
    exit()

data = joblib.load(sys.argv[1], mmap_mode="r")
X = data["X"]
y = data["y"].copy()  # copy is needed to modify y.

y[y >= 1] = 1  # Create a binary classification problem.

clf = SDCAClassifier(tol=1e-5, max_iter=10, verbose=1)
clf.fit(X, y)
print clf.score(X, y)
