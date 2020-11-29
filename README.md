# Parallel Transport Unfolding: Quasi-Isometric Dimensionality Reduction
Parallel Transport Unfolding (PTU) is an algorithm for generating a quasi-isometric, low-dimensional mapping from a sparse and irregular sampling of an arbitrary manifold embedded in a high-dimensional space.

As the name implies, this approach relies on the ideas of parallel transport, metric connection, and Cartan's development. 

The PTU geometric procedure exhibits the same strong resilience to noise as one of the staples of manifold learning, the Isomap algorithm, as it also exploits all pairwise geodesic distances to compute a low-dimensional embedding. While Isomap is limited to geodesically-convex sampled domains, parallel transport unfolding does not suffer from this crippling limitation, resulting in an improved robustness to irregularity and voids in the sampling. Moreover, it involves only simple linear algebra, significantly improves the accuracy of all pairwise geodesic distance approximations, and has the same computational complexity as Isomap.

See the following reference for more details:

**Parallel Transport Unfolding: A Connection-based Manifold Learning Approach.** *Max Budninskiy, Gloria Yin, Leman Feng, Yiying Tong, and Mathieu Desbrun.* SIAM J. Appl. Algebra Geometry, 3(2), pp. 266-291, 2019. ([preprint](http://maxbudninskiy.com/pubs/BYFTD18.pdf))

## Installation
To install dependencies:
```
pip install -r requirements.txt
```
To install the package:
```
python setup.py install
```

## Usage
The core class is `parallel_transport_unfolding.ptu.PTU` is responsible for computing PTU embeddings via method `fit()`. See the following [examples](https://github.com/mbudnins/parallel_transport_unfolding/blob/master/examples/all_examples.ipynb) for usage and comparison with Isomap:

	- embeddings of 2D manifolds living in 3D: 'irregularS', 'holeyS', 'noisySwiss';
	- trivial embedding of a 3D manifold into 3D: 'torus' (spoiler: Isomap fails);
	- embedding of a 2D manifold living in 100D: 'petals'.

In addition, it exposes a separate method `compute_geodesic_distances()` for computing PTU geodesic distance estimates for pointsets sampling low-dimensional manifolds in arbitrary dimensions, with or without input connectivity. The resulting geodesic approximations are significantly more accurate than Dijkstra estimates, and can be useful for other applications. See example 'sphericalCapGeodesics'.

## License
This project is distibuted under modified BSD. Copyright 2020 Max Budninskiy. All rights reserved.

PTU Dijkstra implementation also includes the Fibbonacci heap data structure that was adopted from `scipy.sparse.csgraph._shortest_path.pyx`, developed and owned by Jake Vanderplas under BSD license in 2011.


