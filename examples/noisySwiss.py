from time import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import Isomap

from parallel_transport_unfolding.ptu import PTU
from utils import align, plot_swiss, data_abs_path

# Input 2D manifold embedded in 3D
with open(data_abs_path('./data/noisySwiss/noisySwiss.npy'), 'rb') as f:
    X = np.load(f)
print('Input pointset shape: ', X.shape)

# Non-isometric 2D parametrization of swiss roll. Useful for colormap.
with open(
        data_abs_path('./data/noisySwiss/noisySwiss_2D_nonisometric.npy'),
        'rb') as f:
    p = np.load(f)

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

# Align PTU and Isomap for better visual comparison
iso = align(iso, ptu)

# Plot results
f = plot_swiss(X, ptu, iso, ptu_time, isomap_time, colors=p[:, 0])
plt.show()
