"""!@file EIT_param.py

@brief This code applies the Callaham et al. (2021) method of unsupervised identification of
balance models to the case of Elasto-Inertial Turbulence (EIT) for multiple hyperparameters.
The governing equation here follows the model of a FENE-P fluid in a channel flow. This code makes
the use of the code in the stability_assessment.py script, simply trying multiple hyperparameter
values to compare the results, and make a choice of the best hyperparameter value. This will then
be used in the EIT.py script to identify the balance models.

@details The script performs the following steps:
- Load the data from the Data/ directory, after obtaining it.
- Get the terms as features to represent the data in equation space.
- Cluster the data using Gaussian Mixture Models (GMM) for multiple cluster numbers.
- Apply Sparse Principal Component Analysis (SPCA) to identify the active terms in each cluster.
- Try multiple alpha values to see what dominant balance models are found.

The script takes no arguments

The script outputs the result plots to the 'Plots/EIT/' directory.

@author T. Breitburd, with code from Callaham et al."""

# ----------------------------------------------
# Import modules
# ----------------------------------------------

import h5py
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import SparsePCA
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from joblib import Parallel, delayed


# adding Tools to the system path, and importing the modules
sys.path.insert(0, "../Tools/")
import plot_funcs as pf  # noqa: E402
import stability_assessment as sa  # noqa: E402

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


# Create plot directory
cur_dir = os.getcwd()
proj_dir = os.path.dirname(cur_dir)
plots_dir = os.path.join(proj_dir, "Plots/EIT")
os.makedirs(plots_dir, exist_ok=True)

# ----------------------------------------------
# Load data
# ----------------------------------------------

# Load the data
path = "../Data/39_60_512_512_CAR_Fdp.h5"
data = h5py.File(path, "r")

# Set the values of some parameters
Re = 1000
Wi = 50
beta = 0.9
Lmax = 70
Lx = 2 * np.pi
Ly = 2

# ----------------------------------------------
# Get the equation space representation
# ----------------------------------------------

# Choose the snapshot and trajectory
traj_seed = 5
snapshot = 15

# Extract the data fields
# fmt: off
u = data["u"][
    traj_seed, snapshot - 1:snapshot + 2, :, :]
# Extract the velocity field,
# taking the previous and following
# snapshots as well for time derivatives
# fmt: o

v = data["v"][traj_seed, snapshot, :, :]  # Wall-normal velocity
p = data["p"][traj_seed, snapshot, :, :]  # Pressure field
Fdp = data["Fdp"][traj_seed, snapshot, :, :]  # Time varying pressure gradient
Cxx = data["c11"][
    traj_seed, snapshot, :, :
]  # First diagonal component of the conformation tensor
Cxy = data["c12"][
    traj_seed, snapshot, :, :
]  # Non-diagonal components of the conformation tensor
Cyy = data["c22"][
    traj_seed, snapshot, :, :
]  # Second diagonal component of the conformation tensor
t_prev = data["time"][traj_seed, snapshot - 1]  # Time of the previous snapshot
t = data["time"][traj_seed, snapshot]  # Time of the current snapshot
t_next = data["time"][traj_seed, snapshot + 1]  # Time of the following snapshot

trC = Cxx + Cyy  # Trace of the conformation tensor


# Get the derivatives

# Get the time derivative as 2nd order central difference
dt = t_next - t_prev
u_t = (u[2] - u[0]) / (2 * dt)

# Get the spatial derivatives, using numpy's gradient function
nx = u.shape[2]
ny = u.shape[1]

u_y, u_x = np.gradient(u[1], Ly / ny, Lx / nx)

# The pressure gradient is defined as the sum of the mean pressure gradient
# at the current time step and the x-derivative of the fluctuating pressure term
p_fluc_x = np.gradient(p, Lx / nx, axis=1, edge_order=2)
p_x = Fdp + p_fluc_x


u_xx = np.gradient(u_x, Lx / nx, axis=1, edge_order=2)
u_yy = np.gradient(u_y, Ly / ny, axis=0, edge_order=2)

# Define the RHS terms, separating them in 3 parts.

# The first part is the laplacian of the velocity field
RHS1 = (beta / Re) * (u_xx + u_yy)

# Then are the terms describing the x-component of the divergence of the
# polymer stress tensor: T(C)
RHS2 = (1 / Wi) * ((1 / (1 - (trC - 3) / (Lmax**2))) * Cxx - 1)
RHS3 = (1 / Wi) * ((1 / (1 - (trC - 3) / (Lmax**2))) * Cxy)

RHS2 = ((1 - beta) / Re) * np.gradient(RHS2, Lx / nx, axis=1, edge_order=2)
RHS3 = ((1 - beta) / Re) * np.gradient(RHS3, Ly / ny, axis=0, edge_order=2)


# Define the features:

features = pd.DataFrame(
    {
        "u_t": u_t.flatten(),
        "uu_x": (u[0] * u_x).flatten(),
        "vu_y": (v * u_y).flatten(),
        "p_x": (p_x).flatten(),
        "nondim_Lap_u": (RHS1).flatten(),
        "tensor1": RHS2.flatten(),
        "tensor2": RHS3.flatten(),
    }
)

features = features.to_numpy()
nfeatures = features.shape[1]

# Define the labels
labels = [
    r"$u_{t}$",
    r"$u u_{x}$",
    r"$v u_{y}$",
    r"$p_{x}$",
    r"$\nabla^2 u$",
    r"$[T(C)]_{1}$",
    r"$[T(C)]_{2}$",
]

# ----------------------------------------------
# GMM clustering to choose the number of clusters
# ----------------------------------------------

# Set the random seeds
seed = 75016
np.random.seed(seed)

# Get a subset of the data
frac = 0.3
features_training = train_test_split(features, train_size=frac, random_state=seed)[0]

# Set number of clusters
n_clusters = [3, 5, 7, 9, 11]

for idx, nc in enumerate(n_clusters):
    # Define the model
    GMM = GaussianMixture(n_components=nc, random_state=seed)
    # GMM = CustomGMM(n_components=n_clusters, n_features = 7, random_state=seed)

    # Fit the model
    GMM.fit(features_training)

    # Plot the covariance matrices of each cluster

    pf.plot_cov_mat(
        GMM, features.shape[1], nc, labels, "GMM", f"EIT/GMM_cov_mat_{nc}.png", False
    )

# SET CLUSTER NUMBER TO 8

# Set the random seeds
seed = 75016
np.random.seed(seed)

# Get a subset of the data
frac = 0.3
features_training = train_test_split(features, train_size=frac, random_state=seed)[0]

# Set number of clusters
n_clusters = 8

# Define the model
GMM = GaussianMixture(n_components=n_clusters, random_state=seed)
# GMM = CustomGMM(n_components=n_clusters, n_features = 7, random_state=seed)

# Fit the model
GMM.fit(features_training)

# Predict the clusters for the entire dataset
cluster_idx = GMM.predict(features)


# ----------------------------------------------
# Apply SPCA to choose alpha value
# ----------------------------------------------

alphas = [1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100, 1e3, 1e4, 1e5]
err = np.zeros([len(alphas)])


def spca_err(alpha, cluster_idx, features, nc):
    err_ = 0

    for i in range(nc):
        # Identify points in the field corresponding to each cluster
        feature_idx = np.where(cluster_idx == i)[0]
        cluster_features = features[feature_idx, :]

        # Conduct Sparse PCA
        spca = SparsePCA(n_components=1, alpha=alpha, random_state=seed)
        spca.fit(cluster_features)

        # Identify active and inactive terms
        inactive_terms = np.where(spca.components_[0] == 0)[0]

        err_ += np.sqrt(np.sum((cluster_features[:, inactive_terms].ravel()) ** 2))

    return err_


err = Parallel(n_jobs=4)(
    delayed(spca_err)(alpha, cluster_idx, features, n_clusters) for alpha in alphas
)

pf.plot_spca_residuals(alphas, err, f"EIT/SPCA_residuals_{n_clusters}.png", False)


# Try multiple alpha values to see what dominant balance models are found

# Initialise a list of the number of dominant balance models found
nmodels_list = []

# Define a list of training set sizes, as fractions of the total dataset
alphas = [0.1, 0.2, 0.5, 1, 2, 5, 7, 8, 10, 12, 15, 17]
np.random.seed(75016)

# Plot all found dominant balance regimes for each initial cluster number
plt.figure(figsize=(13, 11))

for idx, alpha in enumerate(alphas):
    # Get the active terms for the current alpha
    spca_model = sa.get_spca_active_terms(alpha, 6, cluster_idx, features, nfeatures)

    (
        balance_models,
        model_index,
        nmodels,
        gridmap,
        grid_labels,
    ) = sa.get_unique_balance_models(spca_model)

    nmodels_list.append(nmodels)

    plt.subplot(3, 4, idx + 1)
    plt.pcolor(
        gridmap, vmin=-0.5, vmax=cm.N - 0.5, cmap=cm, edgecolors="k", linewidth=1
    )
    plt.gca().set_xticks(np.arange(0.5, len(grid_labels) + 0.5))
    plt.gca().set_xticklabels(grid_labels, fontsize=10)
    plt.gca().set_yticklabels([])

    for axis in ["top", "bottom", "left", "right"]:
        plt.gca().spines[axis].set_linewidth(2)

    plt.gca().tick_params(axis="both", width=0)

plot_dir = os.path.join(plots_dir, f"different_alpha_bal_mods_{n_clusters}.png")
plt.savefig(plot_dir, bbox_inches="tight")

plt.close()

plt.figure(figsize=(10, 5))
plt.plot(alphas, nmodels_list, "o-")
plt.xlabel("Alpha value, for the SPCA model")
plt.ylabel("Number of unique \n balance models found")
plt.grid()

plot_dir = os.path.join(plots_dir, f"different_alpha_nmodels_{n_clusters}.png")
plt.savefig(plot_dir, bbox_inches="tight")

plt.close()

# ZOOM IN TO ALPHA VALUES BETWEEN 1 and 4

# Initialise a list of the number of dominant balance models found
nmodels_list = []

# Define a list of training set sizes, as fractions of the total dataset
alphas = [1, 1.5, 2, 2.5, 3, 3.5, 4]
np.random.seed(75016)

# Plot all found dominant balance regimes for each initial cluster number
plt.figure(figsize=(13, 11))

for idx, alpha in enumerate(alphas):
    # Get the active terms for the current alpha
    spca_model = sa.get_spca_active_terms(alpha, 6, cluster_idx, features, nfeatures)

    (
        balance_models,
        model_index,
        nmodels,
        gridmap,
        grid_labels,
    ) = sa.get_unique_balance_models(spca_model)

    nmodels_list.append(nmodels)

    plt.subplot(3, 3, idx + 1)
    plt.pcolor(
        gridmap, vmin=-0.5, vmax=cm.N - 0.5, cmap=cm, edgecolors="k", linewidth=1
    )
    plt.gca().set_xticks(np.arange(0.5, len(grid_labels) + 0.5))
    plt.gca().set_xticklabels(grid_labels, fontsize=10)
    plt.gca().set_yticklabels([])

    for axis in ["top", "bottom", "left", "right"]:
        plt.gca().spines[axis].set_linewidth(2)

    plt.gca().tick_params(axis="both", width=0)

plot_dir = os.path.join(plots_dir, f"different_alpha_bal_mods_zoom_{n_clusters}.png")
plt.savefig(plot_dir, bbox_inches="tight")

plt.close()

plt.figure(figsize=(10, 5))
plt.plot(alphas, nmodels_list, "o-")
plt.xlabel("Alpha value, for the SPCA model")
plt.ylabel("Number of unique \n balance models found")
plt.grid()

plot_dir = os.path.join(plots_dir, f"different_alpha_nmodels_zoom_{n_clusters}.png")
plt.savefig(plot_dir, bbox_inches="tight")

plt.close()
