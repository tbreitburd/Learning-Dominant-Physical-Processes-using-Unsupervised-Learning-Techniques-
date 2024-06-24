# ---------------------------------------------
# Load the modules
# ---------------------------------------------

import sys
import xarray as xr
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from joblib import Parallel, delayed
from sklearn.decomposition import SparsePCA
import sklearn as sk
import os

# adding Tools to the system path, and importing the modules
sys.path.insert(0, "../Tools/")
import plot_funcs as pf  # noqa: E402

# Create the directory for the plots
cur_dir = os.getcwd()
proj_dir = os.path.dirname(cur_dir)
plots_dir = os.path.join(proj_dir, "Plots/Geos_Bal")
os.makedirs(plots_dir, exist_ok=True)

# ---------------------------------------------
# Define the functions
# ---------------------------------------------


def get_derivatives_numpy_geo(nx, ny, x, dy, u, v):
    """Get the derivatives for the 2D domain

    Parameters:
    -----------
    nx: int
        Number of points in the x-direction
    ny: int
        Number of points in the y-direction
    x: numpy array
        Array of x points
    dy: float
        Spacing in the y-direction
    u: numpy array
        Array of u velocities
    v: numpy array
        Array of v velocities


    Returns:
    --------
    u_x: numpy array
        x-derivative of u
    u_y: numpy array
        y-derivative of u
    v_x: numpy array
        x-derivative of v
    v_y: numpy array
        y-derivative of v
    """

    # Initialise the arrays
    u_x = np.zeros((ny, nx))
    u_y = np.zeros((ny, nx))
    v_x = np.zeros((ny, nx))
    v_y = np.zeros((ny, nx))

    # Get the derivatives
    for i in range(ny):
        u_x[i, :] = np.gradient(u[i, :], x[i])
        v_x[i, :] = np.gradient(v[i, :], x[i])

    u_y = np.gradient(u, dy, axis=0)
    v_y = np.gradient(v, dy, axis=0)

    return u_x, u_y, v_x, v_y


# ---------------------------------------------
# Load the data
# ---------------------------------------------

# Read in the data from the nc files at 2 times to get time derivatives

# path1 = "../Data/hycom_gomu_901m000_2019010112_t000.nc"
# path2 = "../Data/hycom_gomu_901m000_2019010112_t001.nc"
path1 = "../Data/hycom_gomu_501_1993010100_t000.nc"
path2 = "../Data/hycom_gomu_501_1993010100_t003.nc"

data_t1 = xr.open_dataset(path1, decode_times=False)
data_t2 = xr.open_dataset(path2, decode_times=False)

# Get the necessary variables
lat = data_t1["lat"].values
lon = data_t1["lon"].values

# Stack the two time snapshots
u = np.array([data_t1["water_u"].values, data_t2["water_u"].values])
u = u[:, 0, 0, :, :]

v = np.array([data_t1["water_v"].values, data_t2["water_v"].values])
v = v[:, 0, 0, :, :]

# Set some domain characteristics
nx = len(lon)
ny = len(lat)
omega = 7.29e-5  # rad/s
f = 2 * omega * np.sin(lat * np.pi / 180)
rho = 1030  # kg/m^3, assuming constant density

# Set the time and space step sizes
R = 6.37e6  # m, Earth radius
dt = 3 * 3600  # 3 hours
# Because dx is variable due to it varying with latitude,
# we calculate it for each latitude
dx = (1 / 25) * (1 / 360) * 2 * np.pi * R * np.cos(lat * np.pi / 180)
x = np.zeros((ny, nx))
for i in range(ny):
    x[i, :] = np.cumsum(np.concatenate(([0], np.repeat(dx[i], nx - 1))))
dy = (1 / 25) * (1 / 360) * 2 * np.pi * R  # Assuming constant Earth radius meriodionaly

print("Data loaded")

# ---------------------------------------------
# Get the derivatives
# ---------------------------------------------

u_x, u_y, v_x, v_y = get_derivatives_numpy_geo(nx, ny, x, dy, u[0], v[0])

# Get the time derivatives
u_t = (u[1] - u[0]) / dt
v_t = (v[1] - v[0]) / dt

# Define the terms
u_grad_u = u[0] * u_x + v[0] * u_y
u_grad_v = u[0] * v_x + v[0] * v_y

# Get the Coriolis term
f = f.reshape(ny, 1)
f_u = -f * u[0]
f_v = f * v[0]

# Get the pressure gradient term
p_x = -rho * (u_t + u_grad_u + f_v)
p_y = -rho * (v_t + u_grad_v + f_u)

print("Derivatives calculated")
# ---------------------------------------------
# Define the features
# ---------------------------------------------

# Flatten the terms
u_t = u_t.flatten()
v_t = v_t.flatten()
u_grad_u = u_grad_u.flatten()
u_grad_v = u_grad_v.flatten()
f_u = f_u.flatten()
f_v = f_v.flatten()
p_x = p_x.flatten()
p_y = p_y.flatten()

# Stack the features for each direction
features_mer = pd.DataFrame(
    {
        "u_t": u_t,
        "u_grad_u": u_grad_u,
        "f_v": f_v,
        "-p_x_over_rho": p_x / rho,
    }
)

features_zon = pd.DataFrame(
    {
        "v_t": v_t,
        "u_grad_v": u_grad_v,
        "f_u": f_u,
        "-p_y_over_rho": p_y / rho,
    }
)

features_mer = features_mer.to_numpy()
features_zon = features_zon.to_numpy()

nfeatures = 4

# Define labels:
labels = [
    r"$\mathbf{u}_{t}$",
    r"$(\mathbf{u} \cdot \nabla)\mathbf{u}$",
    r"$Cor$",
    r"$-\frac{1}{\rho} \nabla p$",
]

# Get rid of nan values, keeping tracks of the indices
mer_nan = np.isnan(features_mer).any(axis=1)
zon_nan = np.isnan(features_zon).any(axis=1)
features_mer = features_mer[~np.isnan(features_mer).any(axis=1)]
features_zon = features_zon[~np.isnan(features_zon).any(axis=1)]

# Combine the meridional and zonal features
features = np.concatenate((features_mer, features_zon), axis=0)

# Scale the features for better GMM performance
features = features * 1e5

print("Features computed")

# ---------------------------------------------
# Gaussian Mixture Model Clustering
# ---------------------------------------------

seed = 75016
np.random.seed(seed)

# Fit the model on 50% of the data
frac = 0.5
features_train, _ = sk.model_selection.train_test_split(
    features, train_size=frac, random_state=seed
)

n_clusters = int(sys.argv[1])
model = GaussianMixture(n_components=n_clusters, random_state=seed)
model.fit(features_train)

# Plot the covariance matrix for each cluster
pf.plot_cov_mat(
    model, nfeatures, n_clusters, labels, "GMM", "Geos_Bal/cov_mat.png", False
)

# Predict clusters for the entire dataset, for a single snapshot
cluster_idx = model.predict(features)


# Get the meridional and zonal points back
n_meridional = len(features_mer[:, 0])

# Initialise the cluster indices
cluster_idx_mer = np.zeros(len(u[0, :, :].flatten()))
cluster_idx_zon = np.zeros(len(u[0, :, :].flatten()))

# Fill the nan values back in
cluster_idx_mer[mer_nan] = np.NaN
cluster_idx_zon[zon_nan] = np.NaN

# Fill the cluster indices
cluster_idx_mer[~mer_nan] = cluster_idx[:n_meridional]
cluster_idx_zon[~zon_nan] = cluster_idx[n_meridional:]

# Reshape the cluster indices
clustermap_mer = cluster_idx_mer.reshape(ny, nx)
clustermap_zon = cluster_idx_zon.reshape(ny, nx)

# Plot the clusters in space
pf.plot_clustering_space_geo(
    clustermap_mer, clustermap_zon, lon, lat, n_clusters, "Geos_Bal/clusters.png", False
)

print("Clusters computed")
# ---------------------------------------------
# Sparse PCA identification of active terms
# ---------------------------------------------

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

pf.plot_spca_residuals(alphas, err, "Geos_Bal/spca_residuals.png", False)

# Set the alpha regularization term to 40
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

pf.plot_balance_models(spca_model, labels, False, "Geos_Bal/active_terms.png", False)

print("Sparse PCA applied")

# ---------------------------------------------
# Get Unique Dominant Balance Models
# ---------------------------------------------

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
    balance_models, labels, False, "Geos_Bal/final_active_terms.png", False
)

# Plot the balance models in a grid
pf.plot_balance_models(
    balance_models, labels, True, "Geos_Bal/balance_models.png", False
)


# Assign the new cluster indices
balance_idx = np.array([model_idx[i] for i in cluster_idx])

n_meridional = len(features_mer[:, 0])

balance_idx_mer = np.zeros(len(u[0].flatten()))
balance_idx_zon = np.zeros(len(u[0].flatten()))

balance_idx_mer[mer_nan] = np.NaN
balance_idx_zon[zon_nan] = np.NaN

balance_idx_mer[~mer_nan] = balance_idx[:n_meridional]
balance_idx_zon[~zon_nan] = balance_idx[n_meridional:]


balancemap_mer = balance_idx_mer.reshape(ny, nx)
balancemap_zon = balance_idx_zon.reshape(ny, nx)

print("Unique balance models computed")

# Plot the balance models in space
pf.plot_clustering_space_geo(
    balancemap_mer,
    balancemap_zon,
    lon,
    lat,
    nmodels,
    "Geos_Bal/balance_models_space.png",
    False,
)
