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


def plot_reynolds_stress(x, y, X, Y, u, Ruv):
    """Plot the Reynolds stress term.

    Args:
    - x: x-coordinates of the grid
    - y: y-coordinates of the grid
    - X: x-coordinates of the grid for the contour plot
    - Y: y-coordinates of the grid for the contour plot
    - u: mean streamwise velocity
    - Ruv: Reynolds stress term
    """

    plt.figure(figsize=(15, 3))
    plt.pcolor(x, y, Ruv, cmap="bone")  # , vmin=0, vmax=1)
    plt.colorbar()

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
    x, y, nx, ny, u, ux, uy, v, Ruvy, Ruux, px, nu, lap_u
):
    """Plot the equation space for the RANS equation terms.

    Args:
    - x: x-coordinates of the grid
    - y: y-coordinates of the grid
    - nx: number of grid points in x-direction
    - ny: number of grid points in y-direction
    - u: mean streamwise velocity
    - ux: streamwise velocity gradient in x-direction
    - uy: streamwise velocity gradient in y-direction
    - v: mean wall-normal velocity
    - Ruvy: Reynolds stress term
    - Ruux: Reynolds stress term
    - px: mean pressure gradient in x-direction
    - nu: kinematic viscosity
    - lap_u: Laplacian of the mean streamwise velocity
    """

    plt.figure(figsize=(15, 5))
    labels = [
        r"$\bar{u} \bar{u}_x$",
        r"$\bar{v}\bar{u}_y$",
        r"$\rho^{-1} \bar{p}_x$",
        r"$\nu \nabla^2 \bar{u}$",
        r"$\overline{(u^\prime v^\prime)}_y$",
        r"$\overline{({u^\prime} ^2)}_x$",
    ]
    clim = 5e-4
    fontsize = 18

    plt.subplot(231)
    field = np.reshape(u * ux, [ny, nx], order="F")
    plt.pcolor(x, y, field, vmin=-clim, vmax=clim, cmap="RdBu")
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.ylabel(labels[0], fontsize=fontsize)

    plt.subplot(232)
    field = np.reshape(v * uy, [ny, nx], order="F")
    plt.pcolor(x, y, field, vmin=-clim, vmax=clim, cmap="RdBu")
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.ylabel(labels[1], fontsize=fontsize)

    plt.subplot(233)
    field = np.reshape(px, [ny, nx], order="F")
    plt.pcolor(x, y, field, vmin=-clim, vmax=clim, cmap="RdBu")
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.ylabel(labels[2], fontsize=fontsize)

    plt.subplot(234)
    field = np.reshape(nu * lap_u, [ny, nx], order="F")
    plt.pcolor(x, y, field, vmin=-clim, vmax=clim, cmap="RdBu")
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.ylabel(labels[3], fontsize=fontsize)

    plt.subplot(235)
    field = np.reshape(Ruvy, [ny, nx], order="F")
    plt.pcolor(x, y, field, vmin=-clim, vmax=clim, cmap="RdBu")
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.ylabel(labels[4], fontsize=fontsize)

    plt.subplot(236)
    field = np.reshape(Ruux, [ny, nx], order="F")
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


def plot_cov_mat(model, nfeatures, nc):
    labels = [
        r"$\bar{u} \bar{u}_x$",
        r"$\bar{v}\bar{u}_y$",
        r"$\rho^{-1} \bar{p}_x$",
        r"$\nu \nabla^2 \bar{u}$",
        r"$\overline{(u^\prime v^\prime)}_y$",
        r"$\overline{({u^\prime} ^2)}_x$",
    ]

    plt.figure(figsize=(12, 15))
    for i in range(nc):
        plt.subplot(3, 3, i + 1)
        C = model.covariances_[i, :, :]
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


def plot_clustering_2d_eq_space(features, cluster_idx, mask, nc):
    labels = [
        r"$\bar{u} \bar{u}_x$",
        r"$\bar{v}\bar{u}_y$",
        r"$\rho^{-1} \bar{p}_x$",
        r"$\nu \nabla^2 \bar{u}$",
        r"$\overline{(u^\prime v^\prime)}_y$",
        r"$\overline{({u^\prime} ^2)}_x$",
    ]

    plt.figure(figsize=(8, 8))
    plt.subplot(221)
    plt.scatter(features[mask, 0], features[mask, 1], 0.1, cluster_idx, cmap=cm)
    plt.xlabel(labels[0], fontsize=20)
    plt.ylabel(labels[1], fontsize=20)
    plt.clim([-0.5, cm.N - 0.5])
    plt.colorbar(boundaries=np.arange(0.5, nc + 1.5), ticks=np.arange(1, nc + 1))
    plt.grid()

    plt.gca().tick_params(axis="both", which="major", labelsize=18)
    plt.gca().tick_params(axis="both", which="minor", labelsize=18)

    plt.subplot(222)
    plt.scatter(features[mask, 1], features[mask, 2], 0.1, cluster_idx, cmap=cm)
    plt.xlabel(labels[1], fontsize=20)
    plt.ylabel(labels[2], fontsize=20)
    plt.clim([-0.5, cm.N - 0.5])
    plt.colorbar(boundaries=np.arange(0.5, nc + 1.5), ticks=np.arange(1, nc + 1))
    plt.grid()

    plt.gca().tick_params(axis="both", which="major", labelsize=18)
    plt.gca().tick_params(axis="both", which="minor", labelsize=18)

    plt.subplot(223)
    plt.scatter(features[mask, 1], features[mask, 3], 0.1, cluster_idx, cmap=cm)
    plt.xlabel(labels[1], fontsize=20)
    plt.ylabel(labels[3], fontsize=20)
    plt.clim([-0.5, cm.N - 0.5])
    plt.colorbar(boundaries=np.arange(0.5, nc + 1.5), ticks=np.arange(1, nc + 1))
    plt.grid()

    plt.gca().tick_params(axis="both", which="major", labelsize=18)
    plt.gca().tick_params(axis="both", which="minor", labelsize=18)

    plt.subplot(224)
    plt.scatter(features[mask, 4], features[mask, 3], 0.1, cluster_idx, cmap=cm)
    plt.xlabel(labels[4], fontsize=20)
    plt.ylabel(labels[3], fontsize=20)
    plt.clim([-0.5, cm.N - 0.5])
    plt.colorbar(boundaries=np.arange(0.5, nc + 1.5), ticks=np.arange(1, nc + 1))
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


def plot_clustering_space(clustermap, x, y, X, Y, nx, ny, nc, u, U_inf):
    plt.figure(figsize=(15, 3))
    plt.pcolor(x, y, clustermap + 1, cmap=cm, vmin=-0.5, vmax=cm.N - 0.5)
    plt.colorbar(boundaries=np.arange(0.5, nc + 1.5), ticks=np.arange(0, nc + 1))

    plt.contour(
        X,
        Y,
        np.reshape(u / U_inf, [ny, nx], order="F"),
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


def plot_spca_residuals(alphas, err):
    plt.figure(figsize=(6, 4))
    plt.scatter(alphas, err)
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
