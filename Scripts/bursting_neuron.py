"""!@file bursting_neuron.py

@brief This code applies the Callaham et al. (2021) method of unsupervised identification of
balance mdoels to the cas of a bursting neuron. The governing equation here is the Generalized
Hodgkin-Huxley model equation, with some added currents. This code makes the use of the code in
the turbulent boundary layer notebook/script, using some of the alternate code. More importantly,
only the information in the paper and paper's supplementary information was used to write this,
alongside the already written code (boundary layer).

@details The script performs the following steps:
- Load the data from the Data/ directory after running the MATLAB data generating code.
- Get the terms as features to represent the data in equation space.
- Cluster the data using Gaussian Mixture Models (GMM).
- Apply Sparse Principal Component Analysis (SPCA) to identify the active terms in each cluster.

This code was written to try and reproduce the results of the paper.

The script takes 2 arguments:
- The number of clusters to use in the GMM, any non-zero positive integer.
- The optimal alpha value for the SPCA, any non-zero positive float.

The script outputs the result plots to the 'Plots/Burst_Neur/' directory.

@author T. Breitburd, with code from Callaham et al."""

# -----------------------------------------
# Import Modules and set plotting parameters
# -----------------------------------------

import scipy.io as sio
import numpy as np
from matplotlib.colors import ListedColormap
import pandas as pd
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import SparsePCA
import sklearn as sk
import sys
from joblib import Parallel, delayed

sys.path.insert(0, "../Tools/")
import plot_funcs as pf  # noqa: E402

# Set the colormap
cm = sns.color_palette("tab10").as_hex()
cm.insert(0, "#ffffff")
cm = ListedColormap(cm)
cm.set_bad("darkgrey")

# -----------------------------------------
# Load Data
# -----------------------------------------

# Load the data
data = sio.loadmat("../Data/burst_data.mat")

# Load the variables
V = data["V"]
t = data["time"]
I_Ca = data["ICa"]
I_CaP = data["ICaP"]
I_SI = data["IISI"]
I_K = data["IK"]
I_L = data["IL"]
I_R = data["IR"]
I_NS = data["INS"]
I_Na = data["INa"]
I_NaCa = data["INaCa"]
I_NaK = data["INaK"]
V_dot = data["dV"]
C_M = data["Cm"]

print("Data loaded")
# -----------------------------------------
# Get the features
# -----------------------------------------

features = pd.DataFrame(
    {
        "RHS": C_M.flatten() * V_dot.flatten(),
        "I_Ca": I_Ca.flatten(),
        "I_CaP": I_CaP.flatten(),
        "I_SI": I_SI.flatten(),
        "I_K": I_K.flatten(),
        "I_L": I_L.flatten(),
        "I_R": I_R.flatten(),
        "I_NS": I_NS.flatten(),
        "I_Na": I_Na.flatten(),
        "I_NaCa": I_NaCa.flatten(),
        "I_NaK": I_NaK.repeat(len(I_NaCa)),
    }
)
nfeatures = features.shape[1]

features = features.to_numpy()

print("Features extracted")
# -----------------------------------------
# Gaussian Mixture Model
# -----------------------------------------
print("Fitting the Gaussian Mixture Model...")
# Set a random seed
seed = 75016
np.random.seed(seed)

# Fit the model on 50% of the data
frac = 0.1
features_train, _ = sk.model_selection.train_test_split(
    features, train_size=frac, random_state=seed
)

n_clusters = int(sys.argv[1])
model = GaussianMixture(n_components=n_clusters, random_state=seed)
model.fit(features_train)

# Define the labels
labels = [
    r"$RHS$",
    r"$\mathbf{I}_{Ca}$",
    r"$\mathbf{I}_{CaP}$",
    r"$\mathbf{I}_{SI}$",
    r"$\mathbf{I}_{K}$",
    r"$\mathbf{I}_{L}$",
    r"$\mathbf{I}_{R}$",
    r"$\mathbf{I}_{NS}$",
    r"$\mathbf{I}_{Na}$",
    r"$\mathbf{I}_{NaCa}$",
    r"$\mathbf{I}_{NaK}$",
]

# Plot the covariance matrices of each cluster
pf.plot_cov_mat(
    model,
    nfeatures,
    n_clusters,
    labels,
    "GMM",
    f"Burst_Neur/cov_mat_{n_clusters}.png",
    False,
)

# Predict the clusters for the rest of the data
cluster_idx = model.predict(features)

# Plot the clusters in the t - V plane
nt = len(t)

# Plot only a subset of the data to get only one of the bursts, as in the paper
idx = np.where((t > 5.5) & (t < 9.5))[0]
t_sub = t[idx]
V_sub = V[idx]
cluster_sub = cluster_idx[idx]

# Plot the clusters
pf.plot_clusters_neuron(
    t_sub, V_sub, cluster_sub, f"Burst_Neur/cluster_{n_clusters}.png", False
)

print("GMM clustering done")
# -----------------------------------------
# Sparse PCA identification of the active terms
# -----------------------------------------

alphas = [1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100, 1e3, 1e4, 1e5]
err = np.zeros([len(alphas)])

print("Finding the optimal alpha value for Sparse PCA...")


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

pf.plot_spca_residuals(
    alphas, err, f"Burst_Neur/spca_residuals_{n_clusters}.png", False
)

print("Applying Sparse PCA for optimal alpha...")
# Set the alpha regularization term to 110
alpha = float(sys.argv[2])
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

pf.plot_balance_models(
    spca_model,
    labels,
    False,
    f"Burst_Neur/active_terms_{n_clusters}_{alpha}.png",
    False,
)


# -----------------------------------------
# Get the unique balance models
# -----------------------------------------
print("Getting the unique balance models...")

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

print("Plotting results...")
# Plot a grid of the active terms
pf.plot_balance_models(
    spca_model,
    labels,
    False,
    f"Burst_Neur/active_terms_{n_clusters}_{alpha}.png",
    False,
)

# Plot the balance models in a grid
pf.plot_balance_models(
    balance_models,
    labels,
    True,
    f"Burst_Neur/final_balance_models_{n_clusters}_{alpha}.png",
    False,
)

# Assign new cluster indices
balance_idx = np.array([model_idx[i] for i in cluster_idx])

# Plot the clusters in the t - V plane
# Get a subset of the data to plot
idx = np.where((t > 5.5) & (t < 9.5))[0]
cluster_sub = balance_idx[idx]

# Plot the clusters
pf.plot_clusters_neuron(
    t_sub,
    V_sub,
    cluster_sub,
    f"Burst_Neur/final_clusters_{n_clusters}_{alpha}.png",
    False,
)

# Plot the clusters for the whole data/timeseries
pf.plot_clusters_neuron(
    t, V, balance_idx, f"Burst_Neur/final_clusters_all_{n_clusters}_{alpha}.png", False
)


# Plot the clusters for 3 projections, as in the paper.
idx = np.where((t > 5.5) & (t < 9.5))[0]
cluster_sub = balance_idx[idx]
I_K_sub = I_K[idx]
RHS_sub = C_M[0] * V_dot[idx]
pf.plot_clusters_neuron_terms(
    RHS_sub,
    I_K_sub,
    "$RHS$",
    "$I_K$",
    cluster_sub,
    f"Burst_Neur/cluster_I_K_{n_clusters}_{alpha}.png",
    False,
)

I_Na_sub = I_Na[idx]
pf.plot_clusters_neuron_terms(
    RHS_sub,
    I_Na_sub,
    "$RHS$",
    "$I_Na$",
    cluster_sub,
    f"Burst_Neur/cluster_I_Na_{n_clusters}_{alpha}.png",
    False,
)

ISI_sub = I_SI[idx]
I_CaP_sub = I_CaP[idx]
pf.plot_clusters_neuron_terms(
    I_CaP_sub,
    ISI_sub,
    "$I_CaP$",
    "$I_SI$",
    cluster_sub,
    f"Burst_Neur/cluster_I_CaP_ISI_{n_clusters}_{alpha}.png",
    False,
)
