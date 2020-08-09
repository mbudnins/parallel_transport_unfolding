from time import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import Isomap

from parallel_transport_unfolding.ptu import PTU
from utils import align, plot_petals, data_abs_path

# Input manifold embedded in 3D, and exact 2D parametrization
with open(data_abs_path('./data/petals/petals_100D.npy'), 'rb') as f:
    X_100D = np.load(f)
with open(data_abs_path('./data/petals/petals_3D.npy'), 'rb') as f:
    X_3D = np.load(f)
print('Input pointset shape: ', X_100D.shape)

# Perform PTU
t = time()
ptu = PTU(
    X=X_100D,
    n_neighbors=10,
    geod_n_neighbors=10,
    embedding_dim=2,
    verbose=True
).fit()
ptu_time = round(time()-t, 2)
print('ptu time: ', ptu_time)

# Perform Isomap
t = time()
iso = Isomap(n_neighbors=10, n_components=2).fit_transform(X_100D)
isomap_time = round(time()-t, 2)
print('isomap time: ', isomap_time)
iso = align(iso, ptu)

# Plot results
f = plot_petals(X_3D, ptu, iso, ptu_time, isomap_time)
plt.show()
