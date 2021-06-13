import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


def data_abs_path(rel_path):
    """
    Determines absolute data path from relative path.

    Parameters
    ----------
    rel_path: string
        Relative path.

    Returns
    ----------
    abs_path: string
        Absolute path.
    """
    abs_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            rel_path
        )
    )
    return abs_path


def center(A):
    """
    Centers matrix A representing N points in d-dim space.

    Parameters
    ----------
    A: matrix
        N x d matrix of N data points

    Returns
    ----------
    Centered matrix representing N points in d-dim space.
    """
    return A - np.mean(A, axis=0)


def align(A, B):
    """
    Aligns matrix A to matrix B with the best orthogonal transform R.
    R = argmin_R ||B - A R||_F

    Parameters
    ----------
    A: matrix
        N x d matrix of N data points to be aligned to B.
    B: matrix
        N x d matrix of N data points.

    Returns
    ----------
    align(A, B): matrix
        Result of the best orthogonal alignment of A to B.
    """
    A = center(A)
    B = center(B)
    U, S, Vt = np.linalg.svd(A.T.dot(B))
    R = U.dot(Vt)
    return A.dot(R)


def pointwise_error(A_hat, A):
    """
    From each of N points of A_hat, computes its normalized distance from
    associated point of pointset A, assuming that points are indexed
    appropriately.
    Distances are normalized by the sqrt of bbox area of A.

    Parameters
    ----------
    A_hat: matrix
        N x d matrix of N data points.
    A: matrix
        N x d matrix of N data points.

    Returns
    ----------
    norm_dists: matrix
        Poinwise relative errors (based on distance) between entries of
        A_hat and A.
    """
    dists = np.linalg.norm(A_hat - A, axis=1)
    bbox_area = (
        (np.max(A[:, 0]) - np.min(A[:, 0])) *
        (np.max(A[:, 1]) - np.min(A[:, 1]))
    )
    norm_dists = dists / np.sqrt(bbox_area)
    return norm_dists


def relative_error(A_hat, A):
    """
    Relative Frobenuis error between matrices A_hat and A.

    Parameters
    ----------
    A_hat: matrix
        N x d matrix of N data points.
    A: matrix
        N x d matrix of N data points.

    Returns
    ----------
    relative_error(A_hat, A): float
        Frobenius relative difference between A_hat and A multiplied by 100.
    """
    return 100 * np.linalg.norm(A_hat - A) / np.linalg.norm(A)


def julia_set(N=300):
    """
    Computes Julia set.

    Parameters
    ----------
    N: int
        Resolution.

    Returns
    ----------
    Julia set of resultion N.
    """
    c = complex(-0.1, 0.65)
    nit_max = 1000
    julia = np.zeros((N, N))
    for ix in range(N):
        for iy in range(N):
            nit = 0
            # Map pixel position to a point in the complex plane
            z = complex(ix / N * 3 - 1.5, iy / N * 3 - 1.5)
            # Do the iterations
            while abs(z) <= 2 and nit < nit_max:
                z = z**2 + c
                nit += 1
            julia[ix, iy] = (nit / nit_max)
    return julia


def plot_S(
        S,
        exact,
        ptu,
        iso,
        ptu_time,
        isomap_time,
        hue='normalized_poinwise_error'):
    """
    Plots 3D embedding of sampled manifold S, its exact 2D parametrization, PTU
    and Isomap embeddings.
    Note that since S is developable, we can compute exact errors.

    Parameters
    ----------
    S: matrix
        N x 3 matrix of sampled S-shaped manifold embedded in 3D.
    exact: matrix
        N x 2 exact isometric parametrization of S (which exists since S is
        developable).
    ptu: matrix
        N x 2 PTU quasi-isometric parametrization of S.
    iso: matrix
        N x 2 Isomap quasi-isometric parametrization of S.
    ptu_time: int
        Time it took for PTU to finish.
    isomap_time: int
        Time it took for Isomap to finish.
    hue: string
        Colormap to use for visualization. Takes two values:
        'normalized_poinwise_error' or 'positional'. Note that we can
        compute pointwise errors, since exact 2D parametrization is known.

    Returns
    ----------
    Figure with plots of 3D embedding of sampled manifold S, its exact 2D
    parametrization, PTU and Isomap embeddings.
    """
    if hue == 'normalized_poinwise_error':
        ptu_hue = pointwise_error(ptu, exact)
        iso_hue = pointwise_error(iso, exact)
        norm = plt.Normalize(
            0,
            0.5 * max(
                np.amax(ptu_hue),
                np.amax(iso_hue)
            )
        )
        cmap = LinearSegmentedColormap.from_list("", ["blue", "red"])
    elif hue == 'positional':
        ptu_hue = exact[:, 0]
        iso_hue = exact[:, 0]
        norm = None
        cmap = 'viridis'
    else:
        raise ValueError('Unrecognized hue for S shaped manifold.')

    f = plt.figure(figsize=(30, 15))

    ax = f.add_subplot(221,  projection='3d', aspect='auto')
    ax.title.set_text('Input 3D data')
    ax.elev = 14
    ax.azim = -70
    ax.scatter(S[:, 0], S[:, 1], S[:, 2], c=exact[:, 0], cmap='viridis', s=50,
               edgecolors='k')

    ax = f.add_subplot(222, aspect='equal')
    ax.title.set_text('Isometric 2D parametrization')
    plt.scatter(exact[:, 0], exact[:, 1], c=exact[:, 0], cmap='viridis')

    ax = f.add_subplot(223, aspect='equal')
    ax.title.set_text('PTU embedding ({} colormap), {}s'
                      .format(hue, ptu_time))
    im = ax.scatter(ptu[:, 0], ptu[:, 1], c=ptu_hue, cmap=cmap, norm=norm)
    plt.colorbar(im, orientation='horizontal')

    ax = f.add_subplot(224, aspect='equal')
    ax.title.set_text('Isomap embedding ({} colormap), {}s'
                      .format(hue, isomap_time))
    im = ax.scatter(iso[:, 0], iso[:, 1], c=iso_hue, cmap=cmap, norm=norm)
    plt.colorbar(im, orientation='horizontal')

    return f


def plot_torus(
        exact,
        ptu,
        iso,
        ptu_time,
        isomap_time,
        hue='normalized_poinwise_error'):
    """
    Plots 3D torus, its 3D PTU embedding, and 3D Isomap embedding.
    Note that since PTU and Isomap map from 3D to 3D, we can compute poinwise
    errors.

    Parameters
    ----------
    exact: matrix
        N x 3 matrix of 3D torus manifold embedded in 3D.
    ptu: matrix
        N x 3 PTU quasi-isometric parametrization of torus.
    iso: matrix
        N x 3 Isomap quasi-isometric parametrization of torus.
    ptu_time: int
        Time it took for PTU to finish.
    isomap_time: int
        Time it took for Isomap to finish.
    hue: string
        Colormap to use for visualization. Takes two values:
        'normalized_poinwise_error' or 'positional'. Note that we can
        compute pointwise errors, since exact 3D parametrization is known.

    Returns
    ----------
    Figure with plots of 3D torus, and its PTU and Isomap embeddings.
    """
    if hue == 'normalized_poinwise_error':
        ptu_hue = pointwise_error(ptu, exact)
        iso_hue = pointwise_error(iso, exact)
        norm = plt.Normalize(
            0,
            0.5 * max(
                np.amax(ptu_hue),
                np.amax(iso_hue)
            )
        )
        cmap = LinearSegmentedColormap.from_list("", ["blue", "red"])
    elif hue == 'positional':
        ptu_hue = exact[:, 1]
        iso_hue = exact[:, 1]
        norm = None
        cmap = 'viridis'
    else:
        raise ValueError('Unrecognized hue for torus shaped manifold.')

    f = plt.figure(figsize=(30, 15))

    ax = f.add_subplot(131,  projection='3d', aspect='auto')
    ax.title.set_text('Input 3D data')
    ax.elev = 70
    ax.azim = 5
    ax.scatter(exact[:, 0], exact[:, 1], exact[:, 2], c=exact[:, 1],
               cmap='viridis', s=50, edgecolors='k')

    ax = f.add_subplot(132,  projection='3d', aspect='auto')
    ax.title.set_text('PTU embedding ({} colormap), {}s'
                      .format(hue, ptu_time))
    ax.elev = 70
    ax.azim = 5
    im = ax.scatter(ptu[:, 0], ptu[:, 1], ptu[:, 2],
                    c=ptu_hue, cmap=cmap, norm=norm, s=50, edgecolors='k')
    plt.colorbar(im, orientation='vertical', fraction=0.046, pad=0.04)

    ax = f.add_subplot(133,  projection='3d', aspect='auto')
    ax.title.set_text('Isomap embedding ({} colormap), {}s'
                      .format(hue, isomap_time))
    ax.elev = 70
    ax.azim = 5
    im = ax.scatter(iso[:, 0], iso[:, 1], iso[:, 2],
                    c=iso_hue, cmap=cmap, norm=norm, s=50, edgecolors='k')
    plt.colorbar(im, orientation='vertical', fraction=0.046, pad=0.04)

    return f


def plot_petals(P, ptu, iso, ptu_time, isomap_time):
    """
    Plots 3D embeding of the petals manifold, its 2D PTU embedding, and 2D
    Isomap embedding.

    Parameters
    ----------
    P: matrix
        N x 3 matrix of 2D petals manifold embedded in 3D.
    ptu: matrix
        N x 2 PTU quasi-isometric embedding of petals.
    iso: matrix
        N x 2 Isomap quasi-isometric embedding of petals.
    ptu_time: int
        Time it took for PTU to finish.
    isomap_time: int
        Time it took for Isomap to finish.

    Returns
    ----------
    Figure with plots of 3D petals, and its PTU and Isomap embeddings.
    """
    f = plt.figure(figsize=(25, 25))

    ax = f.add_subplot(221, aspect='equal')
    ax.title.set_text('Input data is in 100D. Here is Julia set instead.')
    julia = julia_set()
    plt.imshow(julia, cmap='hot')

    ax = f.add_subplot(222,  projection='3d', aspect='auto')
    ax.title.set_text('3D isometric parametrization of input data')
    ax.elev = 14
    ax.azim = -50
    ax.scatter(P[:, 0], P[:, 1], P[:, 2],
               c=P[:, 2], cmap='viridis', s=50, edgecolors='k')

    ax = f.add_subplot(223, aspect='equal')
    ax.title.set_text('2D PTU embedding, {}s'.format(ptu_time))
    ax.scatter(ptu[:, 0], ptu[:, 1], c=P[:, 2], cmap='viridis')

    ax = f.add_subplot(224, aspect='equal')
    ax.title.set_text('2D Isomap embedding, {}s'.format(isomap_time))
    ax.scatter(iso[:, 0], iso[:, 1], c=P[:, 2], cmap='viridis')

    return f


def plot_swiss(X, ptu, iso, ptu_time, isomap_time, colors):
    """
    Plots 3D embedding of the noisy swill roll manifold, its 2D PTU embedding,
    and 2D Isomap embedding.

    Parameters
    ----------
    X: matrix
        N x 3 matrix of 2D swiss roll manifold embedded in 3D.
    ptu: matrix
        N x 2 ptu quasi-isometric embedding of swiss roll.
    iso: matrix
        N x 2 isomap quasi-isometric embedding of swiss roll.
    ptu_time: int
        Time it took for PTU to finish.
    isomap_time: int
        Time it took for Isomap to finish.

    Returns
    ----------
    Figure with plots of 3D swiss roll, and its PTU and Isomap embeddings.
    """
    f = plt.figure(figsize=(30, 20))

    ax = f.add_subplot(131,  projection='3d', aspect='auto')
    ax.title.set_text('Input 3D data')
    ax.elev = 10
    ax.azim = 70
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=colors,
               cmap='viridis', s=50, edgecolors='k')

    ax = f.add_subplot(132, aspect='equal')
    ax.title.set_text('2D PTU embedding, {}s'.format(ptu_time))
    ax.scatter(ptu[:, 0], ptu[:, 1], c=colors, cmap='viridis')

    ax = f.add_subplot(133, aspect='equal')
    ax.title.set_text('2D Isomap embedding, {}s'.format(isomap_time))
    ax.scatter(iso[:, 0], iso[:, 1], c=colors, cmap='viridis')

    return f
