import pylab as plt
import numpy as np
from scipy import misc
from lightning.impl.penalty import TotalVariation2DPenalty

face = misc.imresize(misc.face(gray=True), 0.2)
face = face.astype(np.float) / 255.

# add gaussian noise to the origin image
data = face + 0.2 * np.random.randn(*face.shape)
f, ax = plt.subplots(1, 4, sharey=False)

for i, alpha in enumerate(np.logspace(-2, -0.5, 4)):
    print('Computing inverse problem for alpha=%s' % alpha)
    # clf = FistaRegressor(alpha=alpha, penalty='tv2d', verbose=False, prox_args=face.shape,
    #                      max_iter=1000)
    denoised = TotalVariation2DPenalty(*face.shape).projection([data.ravel()], alpha, 1.0)
    # clf.fit(A, b)
    ax[i].set_title(r'$\alpha$=%.2f' % alpha)
    ax[i].imshow(denoised.reshape(face.shape), interpolation='nearest', cmap=plt.cm.gray)
    ax[i].set_xticks(())
    ax[i].set_yticks(())
plt.show()
