from time import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import Isomap

from parallel_transport_unfolding.ptu import PTU
from utils import align, plot_S, relative_error, data_abs_path

# Input manifold embedded in 3D, and exact 2D parametrization
with open(data_abs_path('./data/irregularS/irregularS.npy'), 'rb') as f:
    X = np.load(f)

with open(
    data_abs_path('./data/irregularS/irregularS_exact_parametrization.npy'),
    'rb'
) as f:
    exact = np.load(f)
print('Input pointset shape: ', X.shape)

# Perform PTU
t = time()
ptu = PTU(
    X=X,
    n_neighbors=10,
    geod_n_neighbors=10,
    embedding_dim=2,
    verbose=True
).fit()
ptu_time = round(time()-t, 2)
print('ptu time: ', ptu_time)

# Perform Isomap
t = time()
iso = Isomap(n_neighbors=10, n_components=2).fit_transform(X)
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
f = plot_S(X, exact, ptu, iso, ptu_time, isomap_time,
           hue='normalized_poinwise_error')
plt.show()
