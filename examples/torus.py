from time import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import Isomap

from parallel_transport_unfolding.ptu import PTU
from utils import align, relative_error, plot_torus, data_abs_path


# Input 3D manifold embedded in 3D
with open(data_abs_path('./data/torus/torus.npy'), 'rb') as f:
    exact = np.load(f)
print('Input pointset shape: ', exact.shape)

# Perform PTU
t = time()
ptu = PTU(
    X=exact,
    n_neighbors=10,
    geod_n_neighbors=10,
    embedding_dim=3,
    verbose=True
).fit()
ptu_time = round(time()-t, 2)
print('ptu time: ', ptu_time)

# Perform Isomap
t = time()
iso = Isomap(n_neighbors=10, n_components=3).fit_transform(exact)
isomap_time = round(time()-t, 2)
print('isomap time: ', isomap_time)

# Align PTU and Isomap to exact parametrization via best isometric
# transformation, and compute errors
ptu = align(ptu, exact)
iso = align(iso, exact)
ptu_error = relative_error(ptu, exact)
iso_error = relative_error(iso, exact)
print('ptu relative error: {}%'.format(ptu_error))
print('isomap relative error: {}%'.format(iso_error))

# Plot results
f = plot_torus(exact, ptu, iso, ptu_time, isomap_time,
               hue='normalized_poinwise_error')
plt.show()
