"""!@bfile plot_funcs.py

@brief Module containing tools for plotting the results of the Boundery-Layer case,
EIT case, and other cases.

@details This module contains tools to plot the results of the Boundary-Layer case,
EIT case, and other cases. Some of the functions are specific to the Boundary-Layer
case, such as the plotting of the Reynolds stress term, the RANS equation terms. But
others can be used for any case, such as the plotting of the covariance matrices
(plot_cov_mat()). The functions are made to be useable for notebooks and scripts,
with the show argument.

@author Created by T.Breitburd on 29/01/2023"""

import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib as mpl
from matplotlib.colors import ListedColormap
import seaborn as sns
from scipy.interpolate import interp1d

# Set the plotting style
mpl.rc("text", usetex=True)
mpl.rc("font", family="serif")
mpl.rc("figure", figsize=(15, 3))
mpl.rc("xtick", labelsize=14)
mpl.rc("ytick", labelsize=14)
mpl.rc("axes", labelsize=20)
mpl.rc("axes", titlesize=20)

# Set the colormap
cm = sns.color_palette("tab10").as_hex()
cm.insert(0, "#ffffff")
cm = ListedColormap(cm)
cm.set_bad("darkgrey")


# Set the labels for the RANS equation terms
labels = [
    r"$\bar{u} \bar{u}_x$",
    r"$\bar{v}\bar{u}_y$",
    r"$\rho^{-1} \bar{p}_x$",
    r"$\nu \nabla^2 \bar{u}$",
    r"$\overline{(u^\prime v^\prime)}_y$",
    r"$\overline{({u^\prime} ^2)}_x$",
]


def plot_reynolds_stress(x, y, X, Y, u, Reynold_stress, path, show=True):
    """Plot the Reynolds stress term, with a line of the 99th percentile
    of the mean streamwise velocity.

    Params:
    - x: x-coordinates of the grid
    - y: y-coordinates of the grid
    - X: x-coordinates of the grid for the contour plot
    - Y: y-coordinates of the grid for the contour plot
    - u: mean streamwise velocity
    - Reynold_uv: Reynolds stress term
    """

    plt.figure(figsize=(10, 4))

    # Plot the Reynolds stress term
    plt.pcolor(x, y, Reynold_stress, cmap="magma")
    cbar = plt.colorbar()
    cbar.set_label(r"$\overline{uv}$ (in $(m.s^{-1})^{2}$)")

    # Plot the 99th percentile of the mean streamwise velocity
    plt.contour(X, Y, u, [0.99], linestyles="dashed", colors="k")
    plt.title(r"Field Plot of Reynold's Stress with 99th percentile of $u$ line")

    plt.tight_layout()

    cur_dir = os.getcwd()
    proj_dir = os.path.dirname(cur_dir)
    plots_dir = os.path.join(proj_dir, "Plots")
    os.makedirs(plots_dir, exist_ok=True)

    plot_dir = os.path.join(plots_dir, path)
    plt.savefig(plot_dir)

    if show:
        plt.show()
    else:
        plt.close()


def plot_equation_terms_bound_lay(
    x,
    y,
    num_x,
    num_y,
    u,
    u_grad_x,
    u_grad_y,
    v,
    Reynold_uv_y,
    Reynold_uu_x,
    p_grad_x,
    nu,
    lap_u,
    path,
    show=True,
):
    """Plot physical space fields of each of the RANS equation terms

    Args:
    - x: x-coordinates of the grid
    - y: y-coordinates of the grid
    - num_x: number of grid points in x-direction
    - num_y: number of grid points in y-direction
    - u: mean streamwise velocity
    - u_grad_x: streamwise velocity gradient in x-direction
    - u_grad_y: streamwise velocity gradient in y-direction
    - v: mean wall-normal velocity
    - Reynold_uv_y: Reynolds stress term
    - Reynold_uu_x: Reynolds stress term
    - p_grad_x: mean pressure gradient in x-direction
    - nu: kinematic viscosity
    - lap_u: Laplacian of the mean streamwise velocity
    """

    plt.figure(figsize=(13, 6))

    global labels
    clim = 5e-4
    fontsize = 18

    # Plot the terms in equation space
    plt.subplot(231)
    if u.ndim == 1 and u_grad_x.ndim == 1:
        field = np.reshape(u * u_grad_x, [num_y, num_x], order="F")
    else:
        field = u * u_grad_x

    plt.pcolor(x, y, field, vmin=-clim, vmax=clim, cmap="RdBu")
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.ylabel(labels[0], fontsize=fontsize)

    plt.subplot(232)
    if v.ndim == 1 and u_grad_y.ndim == 1:
        field = np.reshape(v * u_grad_y, [num_y, num_x], order="F")
    else:
        field = v * u_grad_y
    plt.pcolor(x, y, field, vmin=-clim, vmax=clim, cmap="RdBu")
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.ylabel(labels[1], fontsize=fontsize)

    plt.subplot(233)
    if u.ndim == 1 and p_grad_x.ndim == 1:
        field = np.reshape(u * p_grad_x, [num_y, num_x], order="F")
    else:
        field = u * p_grad_x
    plt.pcolor(x, y, field, vmin=-clim, vmax=clim, cmap="RdBu")
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.ylabel(labels[2], fontsize=fontsize)

    plt.subplot(234)
    if lap_u.ndim == 1:
        field = np.reshape(nu * lap_u, [num_y, num_x], order="F")
    else:
        field = nu * lap_u
    plt.pcolor(x, y, field, vmin=-clim, vmax=clim, cmap="RdBu")
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.ylabel(labels[3], fontsize=fontsize)

    plt.subplot(235)
    if Reynold_uv_y.ndim == 1:
        field = np.reshape(Reynold_uv_y, [num_y, num_x], order="F")
    else:
        field = Reynold_uv_y
    plt.pcolor(x, y, field, vmin=-clim, vmax=clim, cmap="RdBu")
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.ylabel(labels[4], fontsize=fontsize)

    plt.subplot(236)
    if Reynold_uu_x.ndim == 1:
        field = np.reshape(Reynold_uu_x, [num_y, num_x], order="F")
    else:
        field = Reynold_uu_x
    plt.pcolor(x, y, field, vmin=-clim, vmax=clim, cmap="RdBu")
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.ylabel(labels[5], fontsize=fontsize)

    plt.subplots_adjust(
        left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.05
    )

    plt.suptitle(
        "Plots of the values of each RANS equation terms \n in physical space",
        fontsize=20,
    )

    plt.tight_layout()

    field = None

    cur_dir = os.getcwd()
    proj_dir = os.path.dirname(cur_dir)
    plots_dir = os.path.join(proj_dir, "Plots")
    os.makedirs(plots_dir, exist_ok=True)

    plot_dir = os.path.join(plots_dir, path)
    plt.savefig(plot_dir)

    if show:
        plt.show()
    else:
        plt.close()


def plot_cov_mat(model, nfeatures, n_clusters, labels, algorithm, path, show=True):
    """Plot the covariance matrix of the GMM model.

    Args:
    - model: GMM model
    - nfeatures: number of features
    - n_clusters: number of clusters
    - algorithm: algorithm used, can be either 'GMM' or other
    """

    plt.figure(figsize=(12, 10))

    # Get the covariance matrix for each cluster
    for i in range(n_clusters):
        plt.subplot(3, 3, i + 1)
        if algorithm == "GMM":
            C = model.covariances_[i, :, :]
        elif algorithm == "customGMM":
            C = model.covariances[i, :, :]
        else:
            C = model[i, :, :]

        # Plot a colormap of the covariance matrix
        plt.pcolor(
            C, vmin=-max(abs(C.flatten())), vmax=max(abs(C.flatten())), cmap="RdBu"
        )

        plt.gca().set_xticks(np.arange(0.5, nfeatures + 0.5))
        plt.gca().set_xticklabels(labels, fontsize=12)
        plt.gca().set_yticks(np.arange(0.5, nfeatures + 0.5))
        plt.gca().set_yticklabels(labels, fontsize=12)
        plt.gca().set_title("Cluster {0}".format(i))

    plt.subplots_adjust(
        left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.4
    )

    plt.tight_layout()

    cur_dir = os.getcwd()
    proj_dir = os.path.dirname(cur_dir)
    plots_dir = os.path.join(proj_dir, "Plots")
    os.makedirs(plots_dir, exist_ok=True)

    plot_dir = os.path.join(plots_dir, path)
    plt.savefig(plot_dir)

    if show:
        plt.show()
    else:
        plt.close()


def plot_clustering_2d_eq_space(features, cluster_idx, n_clusters, path, show=True):
    """Plot the clustering in the 2D equation space. If the features data is masked, make sure
    that the cluster index is masked as well.

    Args:
    - features: equation space data with terms as features, np.array [n, 6]
    - cluster_idx: cluster labels assigned to each sample point of features, np.array [n]
    - n_clusters: number of clusters, int
    """

    global labels

    plt.figure(figsize=(8, 8))

    # Plot the clustering in the 2D equation space, for different pairs of features
    plt.subplot(221)
    plt.scatter(features[:, 0], features[:, 1], 0.1, cluster_idx, cmap=cm)
    plt.xlabel(labels[0], fontsize=20)
    plt.ylabel(labels[1], fontsize=20)
    plt.clim([-0.5, cm.N - 0.5])
    plt.colorbar(
        boundaries=np.arange(0.5, n_clusters + 1.5), ticks=np.arange(1, n_clusters + 1)
    )
    plt.grid()

    plt.gca().tick_params(axis="both", which="major", labelsize=18)
    plt.gca().tick_params(axis="both", which="minor", labelsize=18)

    plt.subplot(222)
    plt.scatter(features[:, 1], features[:, 2], 0.1, cluster_idx, cmap=cm)
    plt.xlabel(labels[1], fontsize=20)
    plt.ylabel(labels[2], fontsize=20)
    plt.clim([-0.5, cm.N - 0.5])
    plt.colorbar(
        boundaries=np.arange(0.5, n_clusters + 1.5), ticks=np.arange(1, n_clusters + 1)
    )
    plt.grid()

    plt.gca().tick_params(axis="both", which="major", labelsize=18)
    plt.gca().tick_params(axis="both", which="minor", labelsize=18)

    plt.subplot(223)
    plt.scatter(features[:, 1], features[:, 3], 0.1, cluster_idx, cmap=cm)
    plt.xlabel(labels[1], fontsize=20)
    plt.ylabel(labels[3], fontsize=20)
    plt.clim([-0.5, cm.N - 0.5])
    plt.colorbar(
        boundaries=np.arange(0.5, n_clusters + 1.5), ticks=np.arange(1, n_clusters + 1)
    )
    plt.grid()

    plt.gca().tick_params(axis="both", which="major", labelsize=18)
    plt.gca().tick_params(axis="both", which="minor", labelsize=18)

    plt.subplot(224)
    plt.scatter(features[:, 4], features[:, 3], 0.1, cluster_idx, cmap=cm)
    plt.xlabel(labels[4], fontsize=20)
    plt.ylabel(labels[3], fontsize=20)
    plt.clim([-0.5, cm.N - 0.5])
    plt.colorbar(
        boundaries=np.arange(0.5, n_clusters + 1.5), ticks=np.arange(1, n_clusters + 1)
    )
    plt.grid()

    plt.gca().tick_params(axis="both", which="major", labelsize=18)
    plt.gca().tick_params(axis="both", which="minor", labelsize=18)

    plt.subplots_adjust(
        left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.3
    )

    plt.tight_layout()

    cur_dir = os.getcwd()
    proj_dir = os.path.dirname(cur_dir)
    plots_dir = os.path.join(proj_dir, "Plots")
    os.makedirs(plots_dir, exist_ok=True)

    plot_dir = os.path.join(plots_dir, path)
    plt.savefig(plot_dir)

    if show:
        plt.show()
    else:
        plt.close()


def plot_clustering_space(
    clustermap, x, y, X, Y, num_x, num_y, n_clusters, u, U_inf, path, show=True
):
    """Plot the clustering in physical space.

    Args:
    - clustermap: cluster map
    - x: x-coordinates of the grid, np.array [num_x]
    - y: y-coordinates of the grid, np.array [num_y]
    - X: x-coordinates of the grid for the contour plot, np.array [num_y, num_x]
    - Y: y-coordinates of the grid for the contour plot, np.array [num_y, num_x]
    - num_x: number of grid points in x-direction, int
    - num_y: number of grid points in y-direction, int
    - n_clusters: number of clusters, int
    - u: mean streamwise velocity, np.array [num_y, num_x]
    - U_inf: free stream velocity, float
    """

    plt.figure(figsize=(10, 4))
    # Plot the clustering in space
    plt.pcolor(x, y, clustermap + 1, cmap=cm, vmin=-0.5, vmax=cm.N - 0.5)
    plt.colorbar(
        boundaries=np.arange(0.5, n_clusters + 1.5), ticks=np.arange(0, n_clusters + 1)
    )

    # Plot the 99th percentile of the mean streamwise velocity
    plt.contour(
        X,
        Y,
        np.reshape(u / U_inf, [num_y, num_x], order="F"),
        [0.99],
        linestyles="dashed",
        colors="k",
    )

    plt.xlabel("$x$", fontsize=18)
    plt.ylabel("$y$", fontsize=18)
    plt.title("GMM Clusters", fontsize=20)

    plt.tight_layout()

    cur_dir = os.getcwd()
    proj_dir = os.path.dirname(cur_dir)
    plots_dir = os.path.join(proj_dir, "Plots")
    os.makedirs(plots_dir, exist_ok=True)

    plot_dir = os.path.join(plots_dir, path)
    plt.savefig(plot_dir)

    if show:
        plt.show()
    else:
        plt.close()


def plot_active_terms(spca_model, labels, path, show=True):
    """Plot a table of the active terms for each cluster

    Args:
    - spca_model: grid map, np.array [nmodels, nclusters]
    - labels: term labels, list
    """

    # Plot a grid with active terms in each cluster
    gridmap = spca_model.copy()

    # Delete unused terms
    grid_mask = np.nonzero(np.all(gridmap == 0, axis=0))[0]
    gridmap = np.delete(gridmap, grid_mask, axis=1)
    grid_labels = np.delete(labels, grid_mask)

    plt.figure(figsize=(6, 3))

    # Make a grid of the balance models
    plt.pcolor(gridmap, edgecolors="k", linewidth=1, cmap="Greys", vmin=0, vmax=2)
    plt.gca().set_xticks(np.arange(0.5, len(grid_labels) + 0.5))
    plt.gca().set_xticklabels(grid_labels, fontsize=24)
    plt.gca().set_yticklabels([])

    for axis in ["top", "bottom", "left", "right"]:
        plt.gca().spines[axis].set_linewidth(2)

    plt.gca().tick_params(axis="both", width=0)

    plt.xlabel("Terms")
    plt.ylabel("Active terms \n in each cluster")

    plt.tight_layout()

    cur_dir = os.getcwd()
    proj_dir = os.path.dirname(cur_dir)
    plots_dir = os.path.join(proj_dir, "Plots")
    os.makedirs(plots_dir, exist_ok=True)

    plot_dir = os.path.join(plots_dir, path)
    plt.savefig(plot_dir)

    if show:
        plt.show()
    else:
        plt.close()


def plot_spca_residuals(alphas, error, path, show=True):
    """Plot the residuals of the inactive terms in the SPCA model.

    Args:
    - alphas: regularisation term alpha values, list
    - error: SPCA residuals, np.array [len(alphas)]
    """

    plt.figure(figsize=(6, 4))

    # Plot the residuals of the inactive terms
    plt.scatter(alphas, error)

    plt.xlabel(r"$\ell_1$ regularization")
    plt.ylabel("Residual of inactive terms")
    plt.gca().set_xscale("log")
    plt.xlim([1e-5, 1e6])
    plt.grid()

    plt.tight_layout()

    cur_dir = os.getcwd()
    proj_dir = os.path.dirname(cur_dir)
    plots_dir = os.path.join(proj_dir, "Plots")
    os.makedirs(plots_dir, exist_ok=True)

    plot_dir = os.path.join(plots_dir, path)
    plt.savefig(plot_dir)

    if show:
        plt.show()
    else:
        plt.close()


def plot_balance_models(gridmap, grid_labels, path, show=True):
    """Plot a table of the balance models.

    Args:
    - gridmap: grid map, np.array [nmodels, nclusters]
    - grid_labels: term labels with unused terms removed, list
    """

    plt.figure(figsize=(7, 4))

    # Make a grid of the balance models
    plt.pcolor(
        gridmap, vmin=-0.5, vmax=cm.N - 0.5, cmap=cm, edgecolors="k", linewidth=1
    )
    plt.gca().set_xticks(np.arange(0.5, len(grid_labels) + 0.5))
    plt.gca().set_xticklabels(grid_labels, fontsize=24)
    plt.gca().set_yticklabels([])
    # plt.gca().set_yticks(np.arange(0.5, nmodels+0.5))
    # plt.gca().set_yticklabels(range(nc), fontsize=20)
    plt.ylabel("Balance Models")
    plt.xlabel("Terms")

    for axis in ["top", "bottom", "left", "right"]:
        plt.gca().spines[axis].set_linewidth(2)

    plt.gca().tick_params(axis="both", width=0)

    plt.tight_layout()

    cur_dir = os.getcwd()
    proj_dir = os.path.dirname(cur_dir)
    plots_dir = os.path.join(proj_dir, "Plots")
    os.makedirs(plots_dir, exist_ok=True)

    plot_dir = os.path.join(plots_dir, path)
    plt.savefig(plot_dir)

    if show:
        plt.show()
    else:
        plt.close()


def plot_spca_reduced_clustering(x, y, balancemap, path, show=True):
    """Plot the reduced clustering using SPCA.

    Args:
    - x: x-coordinates of the grid, np.array [num_x]
    - y: y-coordinates of the grid, np.array [num_y]
    - balancemap: balance map, np.array [num_y, num_x]
    """

    plt.figure(figsize=(10, 4))

    # Plot the reduced clustering after SPCA
    plt.pcolor(
        x,
        y,
        balancemap + 1,
        cmap=cm,
        vmin=-0.5,
        vmax=cm.N - 0.5,
        alpha=1,
        edgecolors="face",
    )

    plt.gca().set_xticks([])
    plt.gca().set_yticks([])

    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.title("SPCA Reduced Clustering")

    plt.tight_layout()

    cur_dir = os.getcwd()
    proj_dir = os.path.dirname(cur_dir)
    plots_dir = os.path.join(proj_dir, "Plots")
    os.makedirs(plots_dir, exist_ok=True)

    plot_dir = os.path.join(plots_dir, path)
    plt.savefig(plot_dir)

    if show:
        plt.show()
    else:
        plt.close()


def plot_feature_space(features, balance_idx, path, show=True):
    """Plot the points in feature space, coloured according to
    the balance model they belong to. If the features data is masked, make sure
    that the balance index is masked as well.

    Args:
    - features: equation space data with each term as a feature, np.array [n, 6]
    - balance_idx: balance model that the data point belongs to, np.array [n]
    """

    fig, ax = plt.subplots(2, 2, figsize=(8, 8))

    # Plot each balance model in the feature space,
    # in an order that makes the the balance models visible

    c = np.array(cm(balance_idx + 1))

    ax[0, 0].scatter(features[:, 0], features[:, 4], s=1, c=c)
    ax[0, 0].grid()
    ax[0, 1].scatter(features[:, 0], features[:, 1], s=1, c=c)
    ax[0, 1].grid()
    ax[1, 0].scatter(features[:, 0], features[:, 3], s=1, c=c)
    ax[1, 0].grid()
    ax[1, 1].scatter(features[:, 4], features[:, 3], s=1, c=c)
    ax[1, 1].grid()

    ax[0, 0].set_xlabel(labels[0], fontsize=15)
    ax[0, 0].set_ylabel(labels[4], fontsize=15)

    ax[0, 1].set_xlabel(labels[0], fontsize=15)
    ax[0, 1].set_ylabel(labels[1], fontsize=15)

    ax[1, 0].set_xlabel(labels[0], fontsize=15)
    ax[1, 0].set_ylabel(labels[3], fontsize=15)

    ax[1, 1].set_xlabel(labels[4], fontsize=15)
    ax[1, 1].set_ylabel(labels[3], fontsize=15)

    plt.subplots_adjust(
        left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.3
    )
    plt.tight_layout()

    cur_dir = os.getcwd()
    proj_dir = os.path.dirname(cur_dir)
    plots_dir = os.path.join(proj_dir, "Plots")
    os.makedirs(plots_dir, exist_ok=True)

    plot_dir = os.path.join(plots_dir, path)
    plt.savefig(plot_dir)

    if show:
        plt.show()
    else:
        plt.close()


def plot_sublayer_scaling(
    x, y, balancemap, delta, x_layer, gmm_fit, p_gmm, to_fit, path, show=True
):
    """Plot the inertial sublayer scaling.

    Args:
    - x: x-coordinates of the grid
    - y: y-coordinates of the grid
    - balancemap: balance map
    - delta: delta
    - x_layer: x layer
    - gmm_fit: GMM fit
    - p_gmm: GMM parameters
    - x_to_fit: x coordinates over which to fit the inertial balance to the power law
    """

    plt.figure(figsize=(10, 4))

    # Plot the balance model map
    plt.pcolor(x, y, balancemap + 1, cmap=cm, vmin=-0.5, vmax=cm.N - 0.5)

    # Plot the inertial sublayer scaling
    plt.plot(x, delta, "k--", label=r"$0.99 U_\infty$")
    plt.plot(
        x_layer[to_fit],
        gmm_fit[to_fit],
        "k",
        label=r"$\ell \sim x^{{{0:0.2f}}}$".format(p_gmm[1]),
    )
    plt.legend(fontsize=16)

    plt.xlabel("$x$")
    plt.ylabel("$y$")

    plt.title("Inertial sublayer scaling")

    plt.tight_layout()

    cur_dir = os.getcwd()
    proj_dir = os.path.dirname(cur_dir)
    plots_dir = os.path.join(proj_dir, "Plots")
    os.makedirs(plots_dir, exist_ok=True)

    plot_dir = os.path.join(plots_dir, path)
    plt.savefig(plot_dir)

    if show:
        plt.show()
    else:
        plt.close()


def plot_self_similarity(x, visc_bal_idx, y_plus, u_plus, balancemap, path, show=True):
    """Plot the self-similarity of the wall region.

    Args:
    - x: x-coordinates of the grid
    - visc_bal_idx: cluster index of the viscous balance
    - y_plus: y_plus, y-coordinate in wall units
    - u_plus: u_plus, u-velocity in wall units
    - balancemap: array of dominant balance models
    """

    y_fit = []
    u_fit = []
    y_extent = []
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    # For each x-coordinate, plot the near-wall region collapsed profiles
    x_plt = [600, 700, 800, 900, x[-3]]
    for i in range(len(x_plt)):
        x_idx = np.nonzero(x > x_plt[i])[0][0]

        # Find the indices for the viscous balance
        y_idx = np.nonzero(balancemap[:, x_idx] == visc_bal_idx)[0]

        # Print the y+ coordinate where the balance ends (~70)
        y_extent.append(y_plus[y_idx[-1], x_idx])
        print(y_plus[y_idx[-1], x_idx])

        ax1.plot(
            u_plus[:, x_idx],
            y_plus[:, x_idx],
            ".",
            markersize=2,
            label="$x={{{0:d}}}$".format(int(np.round(x[x_idx]))),
        )
        ax2.plot(
            u_plus[y_idx, x_idx],
            y_plus[y_idx, x_idx],
            ".",
            label="$x={{{0:d}}}$".format(int(np.round(x[x_idx]))),
        )

        y_fit = np.concatenate((y_fit, y_plus[y_idx, x_idx]))
        u_fit = np.concatenate((u_fit, u_plus[y_idx, x_idx]))

    y_lim = np.mean(y_extent)
    ax1.plot([0, 30], [y_lim, y_lim], "k--", label="Identified extent")
    ax1.set_xlim([0, 25])
    ax1.legend(fontsize=14, framealpha=1)

    ax1.set_yscale("log")
    ax1.set_ylabel("$y^+$")
    ax1.set_xlabel("$u^+$")
    ax1.set_title("Wall region")
    ax1.grid()

    ax2.set_yscale("log")
    ax2.set_ylabel("$y^+$")
    ax2.set_xlabel("$u^+$")
    ax2.set_title("Wall region (viscous balance only)")

    ax2.grid()

    plt.subplots_adjust(wspace=0.3)

    plt.tight_layout()

    cur_dir = os.getcwd()
    proj_dir = os.path.dirname(cur_dir)
    plots_dir = os.path.join(proj_dir, "Plots")
    os.makedirs(plots_dir, exist_ok=True)

    plot_dir = os.path.join(plots_dir, path)
    plt.savefig(plot_dir)

    if show:
        plt.show()
    else:
        plt.close()


def plot_blasius_solution(eta, f, path, show=True):
    """Plot the Blasius solution.

    Args:
    - eta: similarity parameter
    - f: Blasius solution
    """
    plt.figure(figsize=(4, 4))
    plt.plot(eta, f[:, 0], label="Scipy solution")
    plt.xlabel(r"Similarity parameter $\eta$")
    plt.ylabel(r"$f(\eta)$")
    plt.xlim([0, 6])
    plt.ylim([0, 5])
    plt.grid()
    plt.title("Blasius solution")

    plt.tight_layout()

    cur_dir = os.getcwd()
    proj_dir = os.path.dirname(cur_dir)
    plots_dir = os.path.join(proj_dir, "Plots")
    os.makedirs(plots_dir, exist_ok=True)

    plot_dir = os.path.join(plots_dir, path)
    plt.savefig(plot_dir)

    if show:
        plt.show()
    else:
        plt.close()


def plot_blasius_deviation(x, y, nx, ny, u, eta, f, U_inf, nu, path, show=True):
    """Plot the deviation of the Blasius solution.

    Args:
    - x: x-coordinates of the grid
    - y: y-coordinates of the grid
    - nx: number of grid points in x-direction
    - ny: number of grid points in y-direction
    - u: mean streamwise velocity
    - eta: similarity parameter
    - f: Blasius solution
    - U_inf: free stream velocity
    - nu: kinematic viscosity
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
    u_map = np.reshape(u, [ny, nx], order="F")

    # Create a function of eta for f_prime
    f_prime = interp1d(eta, f[:, 1])  # Second variable from first-order integrator

    for x_plt in [30, 50, 100, 150, 200, 250, 300]:
        x_idx = np.nonzero(x > x_plt)[0][0]

        eta_plot = y * np.sqrt(U_inf / (1.1 * nu * x_plt))

        dev = u_map[:, x_idx] / max(u_map[:, x_idx]) - f_prime(eta_plot)
        if x_plt < 200:
            ax1.plot(u_map[:, x_idx] / max(u_map[:, x_idx]), eta_plot)
            ax2.plot(dev, eta_plot, label="$x={{{0}}}$".format(x_plt))
        else:
            ax1.plot(u_map[:, x_idx] / max(u_map[:, x_idx]), eta_plot, "--")
            ax2.plot(dev, eta_plot, "--", label="$x={{{0}}}$".format(x_plt))

    # plt.ylim([0, 10])
    ax1.plot(f[:, 1], eta, "k", label="Blasius")
    ax1.set_yscale("log")
    ax1.legend(fontsize=16)

    ax1.set_ylabel("$\eta$")  # noqa: W605
    ax1.set_xlabel("$u/U_\infty$")  # noqa: W605
    ax1.grid()

    ax2.legend(fontsize=12)
    ax2.set_yscale("log")

    ax2.set_ylabel("$\eta$")  # noqa: W605
    ax2.set_xlabel("Deviation from Blasius")
    ax2.grid()

    cur_dir = os.getcwd()
    proj_dir = os.path.dirname(cur_dir)
    plots_dir = os.path.join(proj_dir, "Plots")
    os.makedirs(plots_dir, exist_ok=True)

    plot_dir = os.path.join(plots_dir, path)
    plt.savefig(plot_dir)

    if show:
        plt.show()
    else:
        plt.close()


def scatter_clustering_space(x, y, cluster_idx, path, show=True):
    plt.figure(figsize=(15, 5))
    plt.scatter(
        x,
        y,
        c=cluster_idx + 1,
        vmin=-0.5,
        vmax=cm.N - 0.5,
        cmap=cm,
        s=10,
    )
    ax = plt.gca()
    ax.set_facecolor("#000000")

    n_clusters = cluster_idx.max()
    plt.colorbar(
        boundaries=np.arange(0.5, n_clusters + 1.5), ticks=np.arange(1, n_clusters + 1)
    )
    plt.xlabel("$x$", fontsize=18)
    plt.ylabel("$y$", fontsize=18)
    plt.title("Spectral Clustering Clusters", fontsize=20)

    plt.tight_layout()

    cur_dir = os.getcwd()
    proj_dir = os.path.dirname(cur_dir)
    plots_dir = os.path.join(proj_dir, "Plots")
    os.makedirs(plots_dir, exist_ok=True)

    plot_dir = os.path.join(plots_dir, path)
    plt.savefig(plot_dir)

    if show:
        plt.show()
    else:
        plt.close()


def scatter_spca_reduced_clustering(x, y, balance_idx, path, show=True):
    plt.figure(figsize=(15, 5))
    plt.scatter(
        x,
        y,
        c=balance_idx + 1,
        vmin=-0.5,
        vmax=cm.N - 0.5,
        cmap=cm,
        s=10,
    )
    ax = plt.gca()
    ax.set_facecolor("#000000")

    plt.xlabel("$x$", fontsize=18)
    plt.ylabel("$y$", fontsize=18)
    plt.title("SPCA Reduced Clustering", fontsize=20)

    plt.tight_layout()

    cur_dir = os.getcwd()
    proj_dir = os.path.dirname(cur_dir)
    plots_dir = os.path.join(proj_dir, "Plots")
    os.makedirs(plots_dir, exist_ok=True)

    plot_dir = os.path.join(plots_dir, path)
    plt.savefig(plot_dir)

    if show:
        plt.show()
    else:
        plt.close()


def scatter_sublayer_scaling(
    x, x_sc, y_sc, balancemap, delta, x_layer, gmm_fit, x_to_fit, path, show=True
):
    """Plot the inertial sublayer scaling.

    Args:
    - x: x-coordinates of the grid
    - y: y-coordinates of the grid
    - balancemap: balance map
    - delta: delta
    - x_layer: x layer
    - gmm_fit: GMM fit
    - x_to_fit: x coordinates over which to fit the inertial balance to the power law
    """

    plt.figure(figsize=(10, 4))

    # Plot the balance model map
    plt.scatter(x_sc, y_sc, c=balancemap + 1, cmap=cm, vmin=-0.5, vmax=cm.N - 0.5, s=10)
    ax = plt.gca()
    ax.set_facecolor("#000000")
    # Plot the inertial sublayer scaling
    plt.plot(x, delta, "w--", label=r"$0.99 U_\infty$")
    plt.plot(
        x_layer[x_to_fit],
        gmm_fit[x_to_fit],
        "white",
        label=r"$\ell \sim x^{{{0:0.2f}}}$".format(0.80),
    )
    plt.legend(fontsize=16)

    plt.xlabel("$x$")
    plt.ylabel("$y$")

    plt.title("Inertial sublayer scaling")

    plt.tight_layout()

    cur_dir = os.getcwd()
    proj_dir = os.path.dirname(cur_dir)
    plots_dir = os.path.join(proj_dir, "Plots")
    os.makedirs(plots_dir, exist_ok=True)

    plot_dir = os.path.join(plots_dir, path)
    plt.savefig(plot_dir)

    if show:
        plt.show()
    else:
        plt.close()


def plot_clustering_space_geo(
    clustermap_mer, clustermap_zon, x, y, n_clusters, path, show=True
):
    """Plot the clustering for the geostrophic balance in physical space.

    Args:
    - clustermap_mer: cluster map for meridional balance, np.array [num_y, num_x]
    - clustermap_zon: cluster map for zonal balance, np.array [num_y, num_x]
    - x: x-coordinates of the grid, np.array [num_x]
    - y: y-coordinates of the grid, np.array [num_y]
    - n_clusters: number of clusters, int
    """

    plt.figure(figsize=(10, 8))

    plt.subplot(211)
    plt.pcolormesh(x, y, clustermap_mer + 1, cmap=cm, vmin=-0.5, vmax=cm.N - 0.5)
    plt.colorbar(
        boundaries=np.arange(0.5, n_clusters + 1.5), ticks=np.arange(0, n_clusters + 1)
    )
    plt.grid()
    plt.xlabel("Longitude (in degrees)", fontsize=18)
    plt.ylabel("Latitude (in degrees)", fontsize=18)
    plt.title("Meridional Balance", fontsize=20)

    plt.subplot(212)
    plt.pcolormesh(x, y, clustermap_zon + 1, cmap=cm, vmin=-0.5, vmax=cm.N - 0.5)
    plt.colorbar(
        boundaries=np.arange(0.5, n_clusters + 1.5), ticks=np.arange(0, n_clusters + 1)
    )
    plt.grid()
    plt.xlabel("Longitude (in degrees)", fontsize=18)
    plt.ylabel("Latitude (in degrees)", fontsize=18)
    plt.title("Zonal Balance", fontsize=20)

    plt.suptitle("GMM Clusters", fontsize=25)

    plt.tight_layout()

    cur_dir = os.getcwd()
    proj_dir = os.path.dirname(cur_dir)
    plots_dir = os.path.join(proj_dir, "Plots")
    os.makedirs(plots_dir, exist_ok=True)

    plot_dir = os.path.join(plots_dir, path)
    plt.savefig(plot_dir)

    if show:
        plt.show()
    else:
        plt.close()


def plot_clusters_eit(cluster_idx, Lx, nx, Ly, ny, n_clusters, path, show=True):
    """Plot the clustering for the EIT balance in physical space.

    Args:
    - balance_idx: balance index, np.array [ny, nx]
    - Lx: length of the domain in x-direction, float
    - nx: number of grid points in x-direction, int
    - Ly: length of the domain in y-direction, float
    - ny: number of grid points in y-direction, int
    - n_clusters: number of clusters, int
    """

    plt.figure(figsize=(15, 5))

    X, Y = np.meshgrid(np.linspace(0, Lx, nx), np.linspace(-Ly / 2, Ly / 2, ny))

    plt.pcolormesh(X, Y, cluster_idx + 1, cmap=cm, vmin=-0.5, vmax=cm.N - 0.5)
    plt.colorbar(
        boundaries=np.arange(0.5, n_clusters + 1.5), ticks=np.arange(0, n_clusters + 1)
    )

    plt.xlabel("x (in h units)", fontsize=18)
    plt.ylabel("y (in h units)", fontsize=18)
    plt.title("EIT Balance Models", fontsize=20)

    plt.grid()

    plt.tight_layout()

    cur_dir = os.getcwd()
    proj_dir = os.path.dirname(cur_dir)
    plots_dir = os.path.join(proj_dir, "Plots")
    os.makedirs(plots_dir, exist_ok=True)

    plot_dir = os.path.join(plots_dir, path)
    plt.savefig(plot_dir)

    if show:
        plt.show()
    else:
        plt.close()
