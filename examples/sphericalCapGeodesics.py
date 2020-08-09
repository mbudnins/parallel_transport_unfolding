import numpy as np
import scipy.sparse
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.graph_shortest_path import graph_shortest_path

from parallel_transport_unfolding.ptu import PTU
from utils import relative_error, data_abs_path


def special_geodesic_distance(x, y):
    return np.arctan2(np.linalg.norm(np.cross(x, y)), np.dot(x, y))


def spherical_geodesic_dists(X):
    N = X.shape[0]
    dists = np.zeros(shape=(N, N))
    for i in range(0, N):
        for j in range(i+1, N):
            dists[i, j] = special_geodesic_distance(X[i, :], X[j, :])
    dists += dists.T
    return dists


# Input spherical cap pointset
with open(data_abs_path('./data/sphericalCap/sphericalCap.npy'), 'rb') as f:
    X = np.load(f)
print('Input pointset shape: ', X.shape)

# Input triangulation connectivity
T = scipy.sparse.csc_matrix(
        scipy.sparse.load_npz(
            data_abs_path('./data/sphericalCap/triangulation_connectivity.npz')
        )
)

# Computing true geodesic distances on unit sphere
print('Computing true geodesic dists')
true_geod_dists = spherical_geodesic_dists(X)
print('Computing true geodesic dists: done')

# Computing PTU distances
print('Computing PTU dists')
ptu = PTU(
    X=X,
    n_neighbors=10,
    geod_n_neighbors=10,
    embedding_dim=2,
    verbose=True
)
ptu.compute_geodesic_distances()
ptu_dists = ptu.ptu_dists
print('Computing PTU dists: done')

# Computing Dijkstra distances
print('Computing Dijkstra dists')
nn = NearestNeighbors(n_neighbors=10)
nn.fit(X)
graph = nn.kneighbors_graph(mode='distance')
dijkstra_dists = graph_shortest_path(graph, directed=False, method='D')
print('Computing Dijkstra dists: done')

ptu_relative_error = relative_error(ptu_dists, true_geod_dists)
dijkstra_relative_error = relative_error(dijkstra_dists, true_geod_dists)
print('PTU geodesic distances relative error (knn connectivity) = {}%'
      .format(ptu_relative_error))
print('Dijkstra geodesic distances relative error (knn connectivity) = {}%'
      .format(dijkstra_relative_error))

# Computing PTU distances with triangulation connectivity
print('Computing PTU dists with triangulation connectivity')
ptu = PTU.with_custom_graph(
    X=X,
    graph=T,
    geod_n_neighbors=10,
    embedding_dim=2,
    verbose=True
)
ptu.compute_geodesic_distances()
ptu_dists = ptu.ptu_dists
print('Computing PTU dists with triangulation connectivity: done')

# Computing Dijkstra distances with triangulation connectivity
print('Computing Dijkstra dists with triangulation connectivity')
graph = T
dijkstra_dists = graph_shortest_path(graph, directed=False, method='D')
print('Computing Dijkstra dists with triangulation connectivity: done')

ptu_relative_error = relative_error(ptu_dists, true_geod_dists)
dijkstra_relative_error = relative_error(dijkstra_dists, true_geod_dists)
print('PTU geodesic distances relative error (triang. connectivity) = {}%'
      .format(ptu_relative_error))
print('Dijkstra geodesic distances relative error (triang. connectivity)',
      ' = {}%'
      .format(dijkstra_relative_error))
