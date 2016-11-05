"""
Example use of total variation denoising.

We recover an image that was corrupted with gaussian noise.
For this we use the proximal operator of the
total variation penalty (TotalVariation2DPenalty), which solves
a problem of the form

    argmin_x ||corrupted_image - x||^2 + alpha * TV(x)

where TV(x) is the 2-dimensional total variation penalty.
"""
import pylab as plt
import numpy as np
from scipy import misc
from lightning.impl.penalty import TotalVariation2DPenalty

face = misc.imresize(misc.face(gray=True), 0.2)
face = face.astype(np.float) / 255.

# add gaussian noise to the origin image
data = face + 0.2 * np.random.randn(*face.shape)
f, ax = plt.subplots(1, 5, sharey=False)

ax[0].set_title('original')
ax[0].imshow(data, interpolation='nearest', cmap=plt.cm.gray)
ax[0].set_xticks(())
ax[0].set_yticks(())

for i, alpha in enumerate(np.logspace(-1, -0.5, 4)):
    print('Computing denoising for alpha=%s' % alpha)
    denoised = TotalVariation2DPenalty(*face.shape).projection([data.ravel()], alpha, 1.0)
    ax[i+1].set_title(r'$\alpha$=%.2f' % alpha)
    ax[i+1].imshow(denoised.reshape(face.shape), interpolation='nearest', cmap=plt.cm.gray)
    ax[i+1].set_xticks(())
    ax[i+1].set_yticks(())
plt.show()
