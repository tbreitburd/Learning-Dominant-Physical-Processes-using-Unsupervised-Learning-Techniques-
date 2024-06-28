"""!@file EIT.py

@brief Script to apply the callaham et al. (2021) method for unsupervised dominant
balance identification to the EIT dataset.

@details The script loads the EIT dataset, extracts the fields of the velocity,
pressure, and conformation tensor, and computes the spatial and temporal derivatives
of the velocity field. The features are then defined as the terms in the governing
equations of the flow.

The code then GMM clusters the equation-space data, and applies sparse PCA to identify
the active terms and the unique balance models. The balance models are then plotted in a grid
and the clusters are plotted in space.

Finally, the probabilistic nature of the GMM is used to get the uncertainty of the balance models.

This code uses a mix of the alternative code which was written for the turbulent boundary layer
and some of the code inspired by Callaham et al. (2021) code.

@author T. Breitburd on 27/06/2024"""

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

# adding Tools to the system path, and importing the modules
sys.path.insert(0, "../Tools/")
import plot_funcs as pf  # noqa: E402

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
# Define some functions
# ----------------------------------------------


def plot_field(X, Y, field, title, path):
    """
    !@brief Function to plot a field of one of the EIT variables

    @param X: 2D array of x-coordinates
    @param Y: 2D array of y-coordinates
    @param field: 2D array of the field to plot
    """

    plt.figure(figsize=(15, 5))
    plt.pcolor(X, Y, field, cmap="viridis")
    plt.colorbar()
    plt.grid()
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")

    # Save the plot
    cur_dir = os.getcwd()
    proj_dir = os.path.dirname(cur_dir)
    plots_dir = os.path.join(proj_dir, "Plots/EIT")
    os.makedirs(plots_dir, exist_ok=True)

    plot_dir = os.path.join(plots_dir, path)
    plt.savefig(plot_dir)

    plt.close()


def plot_uncertainties(X, Y, uncertainties, path):
    """
    !@brief Function to plot the uncertainties of the balance models

    @param X: 2D array of x-coordinates
    @param Y: 2D array of y-coordinates
    @param uncertainties: 2D array of the uncertainties
    """

    plt.figure(figsize=(10, 5))

    plt.pcolormesh(X, Y, uncertainties, cmap="bone")
    plt.xlabel("x (in h units)")
    plt.ylabel("y (in h units)")
    cbar = plt.colorbar()
    cbar.set_label("Misclassification Probability")

    plt.title("Misclassification Probability of the Balance Models")

    # Save the plot
    cur_dir = os.getcwd()
    proj_dir = os.path.dirname(cur_dir)
    plots_dir = os.path.join(proj_dir, "Plots/EIT")
    os.makedirs(plots_dir, exist_ok=True)

    plot_dir = os.path.join(plots_dir, path)
    plt.savefig(plot_dir)

    plt.close()


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
    traj_seed, snapshot - 1:snapshot + 2, :, :
]  # Extract the velocity field,
# taking the previous and following
# snapshots as well for time derivatives
# fmt: on

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
# at the current time step and the x-derivative of the fluctuating pressure termm
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

# Plot one of the conformation tensor's components:

X, Y = np.meshgrid(np.linspace(0, Lx, nx), np.linspace(-Ly / 2, Ly / 2, ny))
plot_field(X, Y, Cxx, r"$C_{xx}$", path="Cxx.png")


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
# GMM clustering
# ----------------------------------------------

# Set the random seeds
seed = 75016
np.random.seed(seed)

# Get a subset of the data
frac = 0.3
features_training = train_test_split(features, train_size=frac, random_state=seed)[0]

# Set number of clusters
n_clusters = int(sys.argv[1])

# Define the model
GMM = GaussianMixture(n_components=n_clusters, random_state=seed)
# GMM = CustomGMM(n_components=n_clusters, n_features = 7, random_state=seed)

# Fit the model
GMM.fit(features_training)

# Plot the covariance matrices of each cluster
pf.plot_cov_mat(
    GMM,
    features.shape[1],
    n_clusters,
    labels,
    "GMM",
    f"EIT/GMM_cov_mat_{n_clusters}.png",
    False,
)

# Predict the clusters for the entire dataset
cluster_idx = GMM.predict(features)

# FOR UNCERTAINTIES
# Get the cluster membership probabilities
cluster_probs = GMM.predict_proba(features)

# Plot the clusters in space
clustermap = cluster_idx.reshape((ny, nx))
pf.plot_clusters_eit(
    clustermap, Lx, nx, Ly, ny, n_clusters, "EIT/GMM_clusters_{n_clusters}.png", False
)

# ----------------------------------------------
# Apply SPCA
# ----------------------------------------------

# Set the alpha regularization term to its optimal value
alpha = float(sys.argv[2])

# Initialize the sparse PCA model
spca_model = np.zeros((n_clusters, features.shape[1]))

for i in range(n_clusters):
    feature_idx = np.where(cluster_idx == i)[0]
    cluster_features = features[feature_idx, :]

    spca = SparsePCA(n_components=1, alpha=alpha, random_state=seed)
    spca.fit(cluster_features)

    active_terms = np.where(spca.components_[0] != 0)[0]
    if len(active_terms) > 0:
        spca_model[i, active_terms] = 1  # Set the active terms to 1


# Plot the active terms
pf.plot_balance_models(
    spca_model, labels, False, f"EIT/active_terms_{n_clusters}_{alpha}.png", False
)


# ----------------------------------------------
# Get the unique balance models
# ----------------------------------------------

nfeatures = features.shape[1]

# Convert the spca_model array to a dataframe
spca_temp = pd.DataFrame(spca_model.copy())

# Group the balance models by the values of all columns
grouped_models = spca_temp.groupby(np.arange(nfeatures).tolist())
grouped_models = grouped_models.groups.items()

# Combine balance models that have identical active terms
# For each balance model, the spca models that have the same active terms
# are given the same index
balance_models = pd.DataFrame(np.zeros((len(grouped_models), nfeatures)))
model_idx = np.zeros(len(spca_model), dtype=int)

for i, b_model in enumerate(grouped_models):
    idx = b_model[1].to_list()
    model_idx[idx] = i
    balance_models.loc[i] = spca_temp.loc[idx[0]].to_numpy()

# Convert the balance models to a numpy array
balance_models = balance_models.drop_duplicates(keep="first")
balance_models = balance_models.to_numpy()
nmodels = balance_models.shape[0]


# Plot a grid of the active terms
pf.plot_balance_models(
    balance_models,
    labels,
    False,
    f"EIT/final_active_terms_{n_clusters}_{alpha}.png",
    False,
)

# Plot the balance models in a grid
pf.plot_balance_models(
    balance_models, labels, True, f"EIT/balance_models_{n_clusters}_{alpha}.png", False
)

# Assign the new cluster indices
balance_idx = np.array([model_idx[i] for i in cluster_idx])

# Plot the clusters in space
balancemap = balance_idx.reshape((ny, nx))
pf.plot_clusters_eit(
    balancemap,
    Lx,
    nx,
    Ly,
    ny,
    nmodels,
    f"EIT/spca_clustering_space_{n_clusters}_{alpha}.png",
    False,
)


# ----------------------------------------------
# Use GMM's probabilistic nature
# to get the uncertainty of the results
# ----------------------------------------------

# FOR UNCERTAINTIES
# Sum the cluster probabilities for the balance model that the point belongs to
# FOR UNCERTAINTIES
balance_probs = np.zeros((len(cluster_idx)))
for i in range(len(cluster_idx)):
    idx = np.where(model_idx == balance_idx[i])[0]
    balance_probs[i] = 1 - np.sum(cluster_probs[i, idx])

# Plot the balance model probabilities
balance_prob_map = balance_probs.reshape((ny, nx))

plot_uncertainties(
    X, Y, balance_prob_map, path=f"balance_model_uncertainties_{n_clusters}_{alpha}.png"
)
