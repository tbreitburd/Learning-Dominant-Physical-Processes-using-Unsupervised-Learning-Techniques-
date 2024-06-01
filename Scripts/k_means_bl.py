""""""

# ---------------------------------------
# Import modules
# ---------------------------------------

import numpy as np
import h5py
import sys
from sklearn.cluster import KMeans
from sklearn.decomposition import SparsePCA
from scipy.optimize import curve_fit

# adding Tools to the system path
sys.path.insert(0, "../Tools/")
import plot_funcs as pf  # noqa: E402
import preprocessing as pp  # noqa: E402

# ---------------------------------------
# Preprocessing
# ---------------------------------------

# Load the data from http://turbulence.pha.jhu.edu/Transition_bl.aspx
file = h5py.File("../Data/Transition_BL_Time_Averaged_Profiles.h5", "r")

# Get arrays for variables and the Reynold's averages
x = np.array(file["x_coor"])
y = np.array(file["y_coor"])
u_bar = np.array(file["um"])
v_bar = np.array(file["vm"])
p_bar = np.array(file["pm"])
R_uu = np.array(file["uum"]) - u_bar**2
R_uv = np.array(file["uvm"]) - u_bar * v_bar
R_vv = np.array(file["uvm"]) - v_bar**2


# Visualize the wall-normal Reynolds stress
X, Y = np.meshgrid(x, y)

# Define some variables
U_inf = 1
nu = 1 / 800
Re = (U_inf / nu) * x

# ------- Get the derivatives --------
print("----------------------------------")
print("Getting derivatives")
print("----------------------------------")
# Get space steps

dx = x[1] - x[0]
dy = y[1:] - y[:-1]

nx = len(x)  # Number of points in x
ny = len(y)  # Number of points in y

# Get the derivatives
u_x, u_y, lap_u, v_y, p_x, R_uux, R_uvy = pp.get_derivatives_numpy(
    nx, ny, dx, y, u_bar, y, p_bar, R_uu, R_uv
)

# Flatten arrays for matrix multiplication, using fortran ordering
u_bar = u_bar.flatten("F")
v_bar = v_bar.flatten("F")
p_bar = p_bar.flatten("F")
R_uu = R_uu.flatten("F")
R_uv = R_uv.flatten("F")

# Flatten the derivative terms arrays for the rest of the notebook
lap_u = lap_u.flatten("F")
R_uux = R_uux.flatten("F")
R_uvy = R_uvy.flatten("F")
u_x = u_x.flatten("F")
u_y = u_y.flatten("F")
v_y = v_y.flatten("F")
p_x = p_x.flatten("F")

# Labels of terms in the RANS equation
labels = [
    r"$\bar{u} \bar{u}_x$",
    r"$\bar{v}\bar{u}_y$",
    r"$\rho^{-1} \bar{p}_x$",
    r"$\nu \nabla^2 \bar{u}$",
    r"$\overline{(u^\prime v^\prime)}_y$",
    r"$\overline{({u^\prime} ^2)}_x$",
]

# ---------------------------------------------
# Obtain the equation space representation
# of the RANS equation
# ---------------------------------------------

# Gather the terms into an array of features
features = 1e3 * np.vstack([u_bar * u_x, v_bar * u_y, p_x, nu * lap_u, R_uvy, R_uux]).T
nfeatures = features.shape[1]

# ---------------------------------------------
# Cluster using K-Means
# ---------------------------------------------
print("----------------------------------")
print("Clustering with K-Means")
print("----------------------------------")

# Fit Gaussian mixture model
nc = int(sys.argv[1])  # Number of clusters
seed = 76016  # Set a seed for debugging/plotting
model = KMeans(n_clusters=nc, n_init=10, random_state=seed)

# Train on only a subset (10%) of the data due to large size
sample_pct = 0.1
mask = np.random.permutation(features.shape[0])[: int(sample_pct * features.shape[0])]
model.fit(features[mask, :])

# Predict clusters for all data
clustering = model.predict(features)

# Get covariances in each cluster
covs = np.zeros((nc, nfeatures, nfeatures))
for i in range(nc):
    mask_ = clustering == i
    covs[i, :, :] = np.cov(features[mask_, :].T)

# Plot the covariance matrices between terms for each of the K-Means cluster
if nc < 10:
    pf.plot_cov_mat(covs, nfeatures, nc, labels, "Other", "BL/KMeans_CovMat", False)

# ---------------------------------------------
# Cluster the data and visualise:
# equation space and physical space
# ---------------------------------------------

# Predict the cluster index for each data point
cluster_idx = clustering + 1

# Plot the clusters in equation space with 2D projections
pf.plot_clustering_2d_eq_space(
    features, cluster_idx, nc, "BL/KMeans_2D_eq_space.png", False
)

# Assign points in space to each cluster
cluster_idx = clustering
clustermap = np.reshape(cluster_idx, [ny, nx], order="F")

# Visualize the clustering in space
pf.plot_clustering_space(
    clustermap,
    x,
    y,
    X,
    Y,
    nx,
    ny,
    nc,
    u_bar,
    U_inf,
    "BL/KMeans_Clustering_Space.png",
    False,
)

# ---------------------------------------------
# Sparse Principal Component Analysis (SPCA)
# dimensionality reduction
# ---------------------------------------------

print("----------------------------------")
print("Applying Sparse PCA")
print("----------------------------------")
# Sparse PCA to identify directions of nonzero variance in each cluster

alphas = [1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100, 1e3, 1e4, 1e5]
err = np.zeros([len(alphas)])
sparsity = np.zeros([len(alphas)])

for k in range(len(alphas)):
    for i in range(nc):
        # Identify points in the field corresponding to each cluster
        feature_idx = np.nonzero(cluster_idx == i)[0]
        cluster_features = features[feature_idx, :]

        # Conduct Sparse PCA
        spca = SparsePCA(n_components=1, alpha=alphas[k])  # normalize_components=True
        spca.fit(cluster_features)

        # Identify active and terms
        active_terms = np.nonzero(spca.components_[0])[0]
        inactive_terms = [feat for feat in range(nfeatures) if feat not in active_terms]

        # Calculate the error, as the sum of the norms of the inactive terms
        err[k] += np.linalg.norm(cluster_features[:, inactive_terms])

pf.plot_spca_residuals(alphas, err, "BL/KMeans_spca_residuals.png", False)


# Now with optimal alpha, get the active terms in each cluster
alpha_opt = int(sys.argv[2])  # Optimal alpha value

spca_model = np.zeros([nc, nfeatures])  # Store the active terms for each cluster

for i in range(nc):
    feature_idx = np.nonzero(cluster_idx == i)[0]
    cluster_features = features[feature_idx, :]

    spca = SparsePCA(n_components=1, alpha=alpha_opt)  # normalize_components=True
    spca.fit(cluster_features)

    active_terms = np.nonzero(spca.components_[0])[0]
    if len(active_terms) > 0:
        spca_model[i, active_terms] = 1  # Set the active terms to 1

# Plot the active terms in each cluster
pf.plot_balance_models(spca_model, labels, False, "BL/KMeans_active_terms.png", False)


# ---------------------------------------------
# Resulting Final Balance Models
# ---------------------------------------------

print("----------------------------------")
print("Obtaining Final balance models")
print("----------------------------------")

# Identify clusters with identical balance models
balance_models, model_index = np.unique(spca_model, axis=0, return_inverse=True)
nmodels = balance_models.shape[0]

# Make new cluster_idx based on the unique SPCA balance model
balance_idx = np.array([model_index[i] for i in cluster_idx])
balancemap = np.reshape(balance_idx, [ny, nx], order="F")

# Plot the balance models in a grid
pf.plot_balance_models(
    balance_models, labels, True, "BL/KMeans_balance_models.png", False
)

# Plot the clustering in space after SPCA
pf.plot_clustering_space(
    balancemap,
    x,
    y,
    X,
    Y,
    nx,
    ny,
    nmodels,
    u_bar,
    U_inf,
    "BL/KMeans_spca_clustering.png",
    False,
)
# Visualize the clusters in equation space with 2D projections
pf.plot_feature_space(
    features[mask, :], balance_idx[mask], "BL/KMeans_feature_space.png", False
)

# ---------------------------------------------
# Validate the balance models with some diagnostics
# ---------------------------------------------
print("----------------------------------")
print("Validating the balance models")
print("----------------------------------")

# ----- Outer Layer Scaling -----
# The length scale of the outer layer should scale with: l ~ x^(4/5)
print("----- Outer layer scaling -----")

# Create a u_bar field:
u_map = np.reshape(u_bar, (ny, nx), order="F")

# Find which cluster is the inertial sublayer.
inert_sub_idx = np.where(np.all(balance_models == [1, 0, 0, 0, 1, 0], axis=1))[0]


# Define some variables
x_min = 110  # Where inertial balance begins
x_turb = 500  # Where transitional region ends

x_idx = np.nonzero(x > x_min)[0]
x_layer = x[x_idx]

# First, find the upper extent of the inertial sublayer
y_gmm = np.zeros(len(x_idx))
# Loop through wall-normal direction until the balance changes
for i in range(len(x_idx)):
    j = len(y) - 1
    if inert_sub_idx.size == 0:
        y_gmm[i] = y[-1]
        continue
    else:
        while balancemap[j, x_idx[i]] == inert_sub_idx:
            j -= 1
        y_gmm[i] = y[j]  # Store upper value of inertial balance

# Next, find the 99% of free stream velocity line
delta = np.zeros(len(x))
# Loop until velocity falls past 99% freestream
for i in range(len(x)):
    j = 0
    while u_map[j, i] < 0.99:
        j += 1
    delta[i] = y[j - 1]

# Fit inertial balance to power law
power_law = lambda x, a, b: a * x**b  # noqa: E731

x_to_fit = x_layer > x_turb  # End of transitional region
p_gmm, cov = curve_fit(power_law, x_layer[x_to_fit], y_gmm[x_to_fit])
gmm_fit = power_law(x_layer, *p_gmm)
print("Fitted parameters for the power law:")
print(p_gmm)  # Print the fit parameters

# Plot the inertial sublayer scaling
pf.plot_sublayer_scaling(
    x,
    y,
    balancemap,
    delta,
    x_layer,
    gmm_fit,
    p_gmm,
    x_to_fit,
    "BL/KMeans_sublayer_scaling.png",
    False,
)


# ----- Self-similarity -----
# In the near-wall region, the wall-normal profiles of velocity should be self-similar
# over all x locations. Here we find that the identified vertical extent of the viscous
# sublayer is too shallow compared to the "true" vertical extent of the sublayer
# (based on the flow data).
print("----- Self-similarity test -----")

# Compute friction velocity with an estimate of the wall shear stress
u_tau = np.sqrt(nu * u_y[::ny])

# Define wall units
y_plus = np.outer(y, u_tau / nu)
u_plus = np.reshape(u_bar, [ny, nx], order="F") / u_tau

# Plot the self-similarity of the flow
print("y+ coordinates where the balance ends:")
pf.plot_self_similarity(
    x, 0, y_plus, u_plus, balancemap, "BL/KMeans_self_similarity.png", show=False
)
