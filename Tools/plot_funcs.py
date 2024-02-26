"""In this plot_funcs file are defined multiple plotting functions
used in the project."""


import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib as mpl
from matplotlib.colors import ListedColormap
import seaborn as sns

mpl.rc("text", usetex=True)
mpl.rc("font", family="serif")
mpl.rc("figure", figsize=(15, 3))
mpl.rc("xtick", labelsize=14)
mpl.rc("ytick", labelsize=14)
mpl.rc("axes", labelsize=20)
mpl.rc("axes", titlesize=20)

sns_list = sns.color_palette("deep").as_hex()
sns_list.insert(0, "#ffffff")  # Insert white at zero position
sns_cmap = ListedColormap(sns_list)
cm = sns_cmap

mpl_colors = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


labels = [
    r"$\bar{u} \bar{u}_x$",
    r"$\bar{v}\bar{u}_y$",
    r"$\rho^{-1} \bar{p}_x$",
    r"$\nu \nabla^2 \bar{u}$",
    r"$\overline{(u^\prime v^\prime)}_y$",
    r"$\overline{({u^\prime} ^2)}_x$",
]


def plot_reynolds_stress(x, y, X, Y, u, Reynold_uv):
    """Plot the Reynolds stress term.

    Args:
    - x: x-coordinates of the grid
    - y: y-coordinates of the grid
    - X: x-coordinates of the grid for the contour plot
    - Y: y-coordinates of the grid for the contour plot
    - u: mean streamwise velocity
    - Reynold_uv: Reynolds stress term
    """

    plt.figure(figsize=(15, 3))

    # Plot the Reynolds stress term
    plt.pcolor(x, y, Reynold_uv, cmap="bone")  # , vmin=0, vmax=1)
    plt.colorbar()

    # Plot the 99th percentile of the mean streamwise velocity
    plt.contour(X, Y, u, [0.99], linestyles="dashed", colors="k")

    plt.gca().set_yticks([])
    plt.gca().set_xticks([])
    plt.title(r"Reynold's Stress: $\overline{uv}$")

    cur_dir = os.getcwd()
    proj_dir = os.path.dirname(cur_dir)
    plots_dir = os.path.join(proj_dir, "Plots")
    os.makedirs(plots_dir, exist_ok=True)

    plot_dir = os.path.join(plots_dir, "reynolds_stresses.png")
    plt.savefig(plot_dir)

    plt.show()


def plot_equation_space_bound_lay(
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
):
    """Plot the equation space for the RANS equation terms.

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

    plt.figure(figsize=(15, 5))

    global labels
    clim = 5e-4
    fontsize = 18

    # Plot the terms in equation space
    plt.subplot(231)
    field = np.reshape(u * u_grad_x, [num_y, num_x], order="F")
    plt.pcolor(x, y, field, vmin=-clim, vmax=clim, cmap="RdBu")
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.ylabel(labels[0], fontsize=fontsize)

    plt.subplot(232)
    field = np.reshape(v * u_grad_y, [num_y, num_x], order="F")
    plt.pcolor(x, y, field, vmin=-clim, vmax=clim, cmap="RdBu")
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.ylabel(labels[1], fontsize=fontsize)

    plt.subplot(233)
    field = np.reshape(p_grad_x, [num_y, num_x], order="F")
    plt.pcolor(x, y, field, vmin=-clim, vmax=clim, cmap="RdBu")
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.ylabel(labels[2], fontsize=fontsize)

    plt.subplot(234)
    field = np.reshape(nu * lap_u, [num_y, num_x], order="F")
    plt.pcolor(x, y, field, vmin=-clim, vmax=clim, cmap="RdBu")
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.ylabel(labels[3], fontsize=fontsize)

    plt.subplot(235)
    field = np.reshape(Reynold_uv_y, [num_y, num_x], order="F")
    plt.pcolor(x, y, field, vmin=-clim, vmax=clim, cmap="RdBu")
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.ylabel(labels[4], fontsize=fontsize)

    plt.subplot(236)
    field = np.reshape(Reynold_uu_x, [num_y, num_x], order="F")
    plt.pcolor(x, y, field, vmin=-clim, vmax=clim, cmap="RdBu")
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.ylabel(labels[5], fontsize=fontsize)

    plt.subplots_adjust(
        left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.05
    )

    cur_dir = os.getcwd()
    proj_dir = os.path.dirname(cur_dir)
    plots_dir = os.path.join(proj_dir, "Plots")
    os.makedirs(plots_dir, exist_ok=True)

    plot_dir = os.path.join(plots_dir, "equation_space_bound_lay.png")
    plt.savefig(plot_dir)
    plt.show()


def plot_cov_mat(model, nfeatures, n_clusters, algorithm):
    """Plot the covariance matrix of the GMM model.

    Args:
    - model: GMM model
    - nfeatures: number of features
    - n_clusters: number of clusters
    - algorithm: algorithm used, can be either 'GMM' or other
    """

    global labels

    plt.figure(figsize=(12, 15))

    # Get the covariance matrix for each cluster
    for i in range(n_clusters):
        plt.subplot(3, 3, i + 1)
        if algorithm == "GMM":
            C = model.covariances_[i, :, :]
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

    cur_dir = os.getcwd()
    proj_dir = os.path.dirname(cur_dir)
    plots_dir = os.path.join(proj_dir, "Plots")
    os.makedirs(plots_dir, exist_ok=True)

    plot_dir = os.path.join(plots_dir, "cov_mat_bound_lay.png")
    plt.savefig(plot_dir)

    plt.show()


def plot_clustering_2d_eq_space(features, cluster_idx, n_clusters):
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

    cur_dir = os.getcwd()
    proj_dir = os.path.dirname(cur_dir)
    plots_dir = os.path.join(proj_dir, "Plots")
    os.makedirs(plots_dir, exist_ok=True)

    plot_dir = os.path.join(plots_dir, "clustering_2d_eq_space.png")
    plt.savefig(plot_dir)

    plt.show()


def plot_clustering_space(clustermap, x, y, X, Y, num_x, num_y, n_clusters, u, U_inf):
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

    plt.figure(figsize=(15, 3))
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

    cur_dir = os.getcwd()
    proj_dir = os.path.dirname(cur_dir)
    plots_dir = os.path.join(proj_dir, "Plots")
    os.makedirs(plots_dir, exist_ok=True)

    plot_dir = os.path.join(plots_dir, "clustering_space.png")
    plt.savefig(plot_dir)

    plt.show()


def plot_spca_residuals(alphas, error):
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

    cur_dir = os.getcwd()
    proj_dir = os.path.dirname(cur_dir)
    plots_dir = os.path.join(proj_dir, "Plots")
    os.makedirs(plots_dir, exist_ok=True)

    plot_dir = os.path.join(plots_dir, "spca_residuals.png")
    plt.savefig(plot_dir)

    plt.show()


def plot_balance_models(gridmap, grid_labels):
    """Plot a table of the balance models.

    Args:
    - gridmap: grid map, np.array [nmodels, nclusters]
    - grid_labels: term labels with unused terms removed, list
    """

    plt.figure(figsize=(6, 3))

    # Make a grid of the balance models
    plt.pcolor(
        gridmap, vmin=-0.5, vmax=cm.N - 0.5, cmap=cm, edgecolors="k", linewidth=1
    )
    plt.gca().set_xticks(np.arange(0.5, len(grid_labels) + 0.5))
    plt.gca().set_xticklabels(grid_labels, fontsize=24)
    plt.gca().set_yticklabels([])
    # plt.gca().set_yticks(np.arange(0.5, nmodels+0.5))
    # plt.gca().set_yticklabels(range(nc), fontsize=20)
    # plt.ylabel('Balance Model')

    for axis in ["top", "bottom", "left", "right"]:
        plt.gca().spines[axis].set_linewidth(2)

    plt.gca().tick_params(axis="both", width=0)

    cur_dir = os.getcwd()
    proj_dir = os.path.dirname(cur_dir)
    plots_dir = os.path.join(proj_dir, "Plots")
    os.makedirs(plots_dir, exist_ok=True)

    plot_dir = os.path.join(plots_dir, "balance_models.png")
    plt.savefig(plot_dir)
    plt.show()


def plot_spca_reduced_clustering(x, y, balancemap):
    """Plot the reduced clustering using SPCA.

    Args:
    - x: x-coordinates of the grid, np.array [num_x]
    - y: y-coordinates of the grid, np.array [num_y]
    - balancemap: balance map, np.array [num_y, num_x]
    """

    plt.figure(figsize=(15, 3))

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

    # plt.xlabel('$x$')
    # plt.ylabel('$y$')
    plt.title("SPCA Reduced Clustering")

    cur_dir = os.getcwd()
    proj_dir = os.path.dirname(cur_dir)
    plots_dir = os.path.join(proj_dir, "Plots")
    os.makedirs(plots_dir, exist_ok=True)

    plot_dir = os.path.join(plots_dir, "spca_reduced_clustering.png")
    plt.savefig(plot_dir)

    plt.show()


def plot_feature_space(features, balance_idx):
    """Plot the points in feature space, coloured according to
    the balance model they belong to. If the features data is masked, make sure
    that the balance index is masked as well.

    Args:
    - features: equation space data with each term as a feature, np.array [n, 6]
    - balance_idx: balance model that the data point belongs to, np.array [n]
    """

    fontsize = 20
    size = 1

    # Plot order of the terms for best visibility
    order = [3, 0, 4, 1, 2]

    fig, ax = plt.subplots(2, 2, figsize=(8, 8))

    # Plot each balance model in the feature space,
    # in an order that makes the the balance models visible
    for k in order:
        plt_ = np.nonzero(balance_idx[:] == k)
        c = np.array(cm(k + 1))[None, :]
        ax[0, 0].scatter(features[plt_, 0], features[plt_, 4], s=size, c=c)
        ax[0, 1].scatter(features[plt_, 0], features[plt_, 1], s=size, c=c)
        ax[1, 0].scatter(features[plt_, 0], features[plt_, 3], s=size, c=c)
        ax[1, 1].scatter(features[plt_, 4], features[plt_, 3], s=size, c=c)

    ax[0, 0].set_xlabel(labels[0], fontsize=fontsize)
    ax[0, 0].set_ylabel(labels[4], fontsize=fontsize)

    ax[0, 1].set_xlabel(labels[0], fontsize=fontsize)
    ax[0, 1].set_ylabel(labels[1], fontsize=fontsize)

    ax[1, 0].set_xlabel(labels[0], fontsize=fontsize)
    ax[1, 0].set_ylabel(labels[3], fontsize=fontsize)

    ax[1, 1].set_xlabel(labels[4], fontsize=fontsize)
    ax[1, 1].set_ylabel(labels[3], fontsize=fontsize)

    for i in [0, 1]:
        for j in [0, 1]:
            ax[i, j].grid()

            # ax[i,j].set_xticklabels([])
            # ax[i,j].set_yticklabels([])
            # ax[i,j].tick_params(axis='x', length=10, width=0)
            # ax[i,j].tick_params(axis='y', length=0)

    plt.subplots_adjust(
        left=None, bottom=None, right=None, top=None, wspace=0.3, hspace=0.3
    )

    cur_dir = os.getcwd()
    proj_dir = os.path.dirname(cur_dir)
    plots_dir = os.path.join(proj_dir, "Plots")
    os.makedirs(plots_dir, exist_ok=True)

    plot_dir = os.path.join(plots_dir, "feature_space.png")
    plt.savefig(plot_dir)

    plt.show()


def plot_sublayer_scaling(x, y, balancemap, delta, x_layer, gmm_fit, p_gmm, to_fit):
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

    plt.figure(figsize=(15, 3))

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
    plt.show()


def plot_self_similarity(x, y_plus, u_plus, balancemap):
    """Plot the self-similarity of the wall region.

    Args:
    - x: x-coordinates of the grid
    - y_plus: y_plus, y-coordinate in wall units
    - u_plus: u_plus, u-velocity in wall units
    - balancemap: array of dominant balance models
    """

    y_fit = []
    u_fit = []

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    x_plt = [600, 700, 800, 900, x[-3]]

    # For each x-coordinate, plot the near-wall region collapsed profiles
    for i in range(len(x_plt)):
        x_idx = np.nonzero(x > x_plt[i])[0][0]

        # Find the indices for the viscous balance
        y_idx = np.nonzero(balancemap[:, x_idx] == 0)[0]

        # Print the y+ coordinate where the balance ends (~70)
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

    ax1.plot([0, 30], [70, 70], "k--", label="Identified extent")
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
    plt.show()


def plot_blasius_solution(eta, f):
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
    plt.show()
