# --------------------------------------------
# Import modules
# --------------------------------------------

import scipy.io as sio
import sys
import os
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import SparsePCA
import matplotlib.pyplot as plt
import sklearn as sk
from joblib import Parallel, delayed

# adding Tools to the system path, and importing the modules
sys.path.insert(0, "../Tools/")
import plot_funcs as pf  # noqa: E402

cur_dir = os.getcwd()
proj_dir = os.path.dirname(cur_dir)
plots_dir = os.path.join(proj_dir, "Plots/Opt_Pul")
os.makedirs(plots_dir, exist_ok=True)

# --------------------------------------------
# Define any functions
# --------------------------------------------


def plot_field(x, t, field, path):
    """!@brief Plot the field in space and time

    @param x: The space coordinate
    @param t: The time coordinate
    @param field: The field to plot
    @param path: The path to save the plot
    """

    fig = plt.figure(figsize=(7, 6))  # noqa: F841
    max_u = np.max(field)
    plt.pcolormesh(t, x, field, vmin=max_u - 40, vmax=max_u)
    plt.xlim([-50, 1500])
    cbar = plt.colorbar()
    cbar.set_label("Intensity [dB]")
    plt.xlabel("Time (in picoseconds)")
    plt.ylabel("Distance (in meters)")
    plt.title("u (in dB)")

    # Save the plot
    cur_dir = os.getcwd()
    proj_dir = os.path.dirname(cur_dir)
    plots_dir = os.path.join(proj_dir, "Plots")
    os.makedirs(plots_dir, exist_ok=True)

    plot_dir = os.path.join(plots_dir, path)
    plt.savefig(plot_dir)

    plt.close()


def plot_term_projections(features, labels):
    """!@brief Plot the projections of the features

    @param features: The features to plot
    @param labels: The labels of the features
    """

    # Visualise some of the features
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))  # noqa: F841

    # Plot the features
    ax[0].scatter(features[:, 0], features[:, 1], s=1)
    ax[0].set_xlabel(labels[0])
    ax[0].set_ylabel(labels[1])

    ax[1].scatter(features[:, 6], features[:, 7], s=1)
    ax[1].set_xlabel(labels[6])
    ax[1].set_ylabel(labels[7])

    # Save the plot
    cur_dir = os.getcwd()
    proj_dir = os.path.dirname(cur_dir)
    plots_dir = os.path.join(proj_dir, "Plots")
    os.makedirs(plots_dir, exist_ok=True)

    plot_dir = os.path.join(plots_dir, "Opt_Pul/features.png")
    plt.savefig(plot_dir)

    plt.close()


# --------------------------------------------
# Load Data
# --------------------------------------------

# Load the data
data = sio.loadmat("../Data/gnlse_nondim.mat")

# Extract the variables needed:
u = data["u"]
u_x = data["ux"]
u_t = data["du_dt"][0, 1]
u_tt = data["du_dt"][0, 2]
u_3t = data["du_dt"][0, 3]
u_4t = data["du_dt"][0, 4]
u_5t = data["du_dt"][0, 5]
u_6t = data["du_dt"][0, 6]

R = data["raman"]
u2u = data["kerr"]

# Visualise the terms in space
x = data["x"]
t = data["t"]

# Plot one of the terms: u
field = 10 * np.log10(np.abs(u) ** 2)

plot_field(x, t, field, "Opt_Pul/u.png")

# --------------------------------------------
# Get the features
# --------------------------------------------

# Define the labels
labels = [
    r"$u_{t}^{(2)}$",
    r"$u_{t}^{(3)}$",
    r"$u_{t}^{(4)}$",
    r"$u_{t}^{(5)}$",
    r"$u_{t}^{(6)}$",
    r"$|u|^{2}$",
    r"$R$",
    r"$u_{x}$",
]

# Get the terms in an array
# Because it is of little interest, remove all points for t < 200
idx = np.where(t > -200)[1][0]

u_x_sub = u_x[:, idx:]
u_tt_sub = u_tt[:, idx:]
u_3t_sub = u_3t[:, idx:]
u_4t_sub = u_4t[:, idx:]
u_5t_sub = u_5t[:, idx:]
u_6t_sub = u_6t[:, idx:]
R_sub = R[:, idx:]
u2u_sub = u2u[:, idx:]

# Flatten the terms
u_x_sub = u_x_sub.flatten()
u_tt_sub = u_tt_sub.flatten()
u_3t_sub = u_3t_sub.flatten()
u_4t_sub = u_4t_sub.flatten()
u_5t_sub = u_5t_sub.flatten()
u_6t_sub = u_6t_sub.flatten()
R_sub = R_sub.flatten()
u2u_sub = u2u_sub.flatten()

# Create the array according to Fig 9 in the supplementary informations
features = np.array(
    np.array(
        [u_tt_sub, u_3t_sub, u_4t_sub, u_5t_sub, u_6t_sub, u2u_sub, R_sub, u_x_sub]
    ).T
)
nfeatures = features.shape[1]

# Because complex values are not supported in the GMM,
# we will use the real part of the features for now
features = np.real(features)

# Visualise some of the features
plot_term_projections(features, labels)

# --------------------------------------------
# GMM clustering
# --------------------------------------------

# Set the seed
seed = 75016
np.random.seed(seed)

# Fit the model on 50% of the data
frac = 0.15
features_train, _ = sk.model_selection.train_test_split(
    features, train_size=frac, random_state=seed
)

n_clusters = 6
model = GaussianMixture(n_components=n_clusters, random_state=seed)
model.fit(features_train)

# Plot the covariance matrices of each cluster
pf.plot_cov_mat(
    model, nfeatures, n_clusters, labels, "GMM", "Opt_Pul/cov_mat.png", False
)

# Predict the clusters for the entire dataset
cluster_idx = model.predict(features)

# Define a clustermap by reshaping the cluster_idx
nx = len(x.flatten())
nt = len(t[0, idx:].flatten())

clustermap = cluster_idx.reshape(nx, nt)

# Plot the clustermap
t_sub = t[:, idx:]

pf.plot_clustering_optics(
    clustermap, x, t_sub, n_clusters, "Opt_Pul/clustermap.png", False
)

# --------------------------------------------
# Sparse PCA identification of active terms
# --------------------------------------------

alphas = [1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100, 1e3, 1e4, 1e5]
err = np.zeros([len(alphas)])


def spca_err(alpha, cluster_idx, features, nc):
    err_ = 0

    for i in range(nc):
        # Identify points in the field corresponding to each cluster
        feature_idx = np.where(cluster_idx == i)[0]
        cluster_features = features[feature_idx, :]

        # Conduct Sparse PCA
        spca = SparsePCA(n_components=1, alpha=alpha)
        spca.fit(cluster_features)

        # Identify active and inactive terms
        inactive_terms = np.where(spca.components_[0] == 0)[0]

        err_ += np.sqrt(np.sum((cluster_features[:, inactive_terms].ravel()) ** 2))

    return err_


err = Parallel(n_jobs=4)(
    delayed(spca_err)(alpha, cluster_idx, features, n_clusters) for alpha in alphas
)

pf.plot_spca_residuals(alphas, err, "Opt_Pul/spca_residuals.png", False)

# Set the alpha regularization term to 10
alpha = 10

# Initialize the sparse PCA model
spca_model = np.zeros((n_clusters, nfeatures))

for i in range(n_clusters):
    feature_idx = np.where(cluster_idx == i)[0]
    cluster_features = features[feature_idx, :]

    spca = SparsePCA(n_components=1, alpha=alpha, random_state=seed)
    spca.fit(cluster_features)

    active_terms = np.where(spca.components_[0] != 0)[0]
    if len(active_terms) > 0:
        spca_model[i, active_terms] = 1  # Set the active terms to 1

pf.plot_balance_models(spca_model, labels, False, "Opt_Pul/active_terms.png", False)


# --------------------------------------------
# Get the unique dominant balance models
# --------------------------------------------

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
for i, model in enumerate(grouped_models):
    idx = model[1].to_list()
    model_idx[idx] = i
    balance_models.loc[i] = spca_temp.loc[idx[0]].to_numpy()

# Convert the balance models to a numpy array
balance_models = balance_models.drop_duplicates(keep="first")
balance_models = balance_models.to_numpy()
nmodels = balance_models.shape[0]

# Plot a grid of the active terms
pf.plot_balance_models(
    balance_models, labels, False, "Opt_Pul/final_active_terms.png", False
)

# Plot the balance models in a grid
pf.plot_balance_models(
    balance_models, labels, True, "Opt_Pul/balance_models.png", False
)

# Assign the new cluster indices
balance_idx = np.array([model_idx[i] for i in cluster_idx])

# Define a new clustermap
balance_clustermap = balance_idx.reshape(nx, nt)

pf.plot_clustering_optics(
    balance_clustermap, x, t_sub, nmodels, "Opt_Pul/balance_clustermap.png", False
)

# Plot the 3D plot of the balance models
pf.plot_optical_pulse_3D(
    x, t, field, balance_clustermap, "Opt_Pul/balance_models_3D.png", False
)
