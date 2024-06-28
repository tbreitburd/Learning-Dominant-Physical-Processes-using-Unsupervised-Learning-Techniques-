# DIS Project 24: Learning Dominant Physical Processes with data driven balance models



# Summary

This repository contains the code for the Research Project completed as partial fulfilment of the MPhil Data Intensive Science.

This project explores a relatively recent approach to identify dominant physical processes in varied physical systems using data-driven balance models. By leveraging machine learning techniques such as Gaussian Mixture Model clustering and Sparse Principal Component Analysis, the Callaham et al. (2021) method aims to automate the simplification of equations that govern physical phenomena, and hence their simulation.

In short, Callaham et al. (2021) introduced a three-step data-driven method to identify dominant balance models in physical systems. The methodology consists of:

1.	Data & Equation Space Representation: Physical system data (e.g., velocity, pressure) are transformed into an equation space where each term in the governing equations forms a dimension, creating a feature space in which clustering is run.
2.	Gaussian Mixture Model (GMM) Clustering (or other algorithms): Thanks to the equation space representation, unsupervised clustering will mean grouping points which have a similar balance of terms. This step requires setting a hyperparameter: the number of clusters, which determines the number of Gaussian distributions assumed in the data.
3.	Sparse Principal Component Analysis (SPCA): SPCA is then applied to each cluster, identifying the active terms by applying a sparsity constraint. This constraint helps in reducing the number of non-zero coefficients, simplifying the model but keeping the most information possible. This sparsity constraint's intensity is set by a second hyperparameter: the alpha (α) parameter, the larger it is, the more terms will be considered inactive in a cluster.

One can adjust the number of clusters for the GMM and the alpha value for SPCA to optimize the method for different datasets and desired levels of model complexity.

The main goal of the project is to discuss the reproducibility of the results, using alternative code. It also delves into thoroughly testing the robustness of the methods, by testing other clustering algorithms, as well as testing the stability of the results when changing hyperparameters.

Its secondary goal was to try using that original method on some novel data, testing the ability of the model to generalise to not well-studied physical systems, and learning more about that type of flow.

In this repository are notebooks for every physical system studied, and algorithm tested (all found in the ```Notebooks/``` directory). For each notebook, there is a script version that is there for portability and if one wants to test other values of hyperparameters (all found in the ```Scripts/``` directory).

# Contents

As this project is quite large, a detailed description of its contents is done below.

## Data

In this directory should be stored the data for any physical system studied in this project. For most of them the data was obtained from external sources. There is an exception which is for the Elasto-Inertial Turbulence. The data for this novel case was obtained internally through the DAMTP, simulated using the Dedalus framework ??

### Turbulent Boundary Layer

The data is a Direct Numerical Simulation (DNS) of a boundary layer in transition to turbulence by [Lee and Zaki](https://turbulence.pha.jhu.edu/docs/README-transition_bl.pdf) and is available on the John Hopkins Turbulence Databases: [here](https://turbulence.pha.jhu.edu/Transition_bl.aspx)

### Geostrophic Balance

The data here is obtained from the high-resolution 1/$25^{o}$ [HYCOM](https://www.hycom.org/hycom) reanalysis data from the Gulf of Mexico. The paper states that only the first field from Exoeriment 50.1 is used (i.e. January 1993), and it is what is used to get the results presented in the report. However, the results obtained in the Callaham et al, (2021) paper were actually obtained using Expermiment 90.1m000 which had its first fields in 2019.

Experiment 50.1, first 1993 fields: [here](https://data.hycom.org/datasets/GOMu0.04/expt_50.1/data/netcdf/1993/)

Experiment 90.1m000, first 2019 fields: [here](https://data.hycom.org/datasets/GOMu0.04/expt_90.1m000/data/hindcasts/2019/)

In both cases, the first and second field available were downloaded in order get the time derivative terms.

----
For the data that cannot be downloaded from an external source, there is data generating code in 2 subdirectories: ```gnlse/``` and ```neuron/```.

### Generalised NonLinear Schrödinger Equations (GNLSE)

Data generating code for the optical pulse case. See subdirectory ```README```.

### Generalized Hodgkin-Huxley Model of bursting neuron

Data generating code for the bursting neuron case. See subdirectory ```README```.



## Notebooks & Scripts

### Turbulent Boundary Layer Case

The main case that is studied is the case of a boundary layer in transition to turbulence. Beyond using the Callaham et al. (2021) method to identify the dominant balance models for this physical system, this notebook explores the reproducing of results using alternative code to make sure the paper's results are not unique and random.

The governing equation here is the Reynold's Averaged Navier-Stokes Equation, and only the streamwise component is considered:

$$ \bar{u} \bar{u}_x + \bar{v} \bar{u}_y = \rho^{-1} \bar{p}_x + \nu \nabla^2 \bar{u}  - (\overline{u' v'})_y - (\overline{u'^2})_x $$

To choose between running the code for the original or custom version, there is a cell dedicated to this where one can comment out/uncomment the ```method``` defining lines. From there the whole Notebook should run without input. The value for the hyperparameters (cluster number, and $\alpha$) are set for the notebook, though they can of course be changed.

More importantly, the notebook is there for presenting the code in a more guided manner but there is a script which ensures portability, faster runtime, and the ability for the user to control the hyperparameter values, as well as the method used (original or alternate)

The script is run as follows:
```bash
$ cd Scripts/
$ python boundary_layer.py <method> <cluster number> <alpha>
```

Where ```<method>``` must be a string, either ```'original'``` or any other string which will count as the other method and run the alternative code. ```<cluster number>``` can be any non-zero integer, though values larger than 15 are likely excessive and will not bring better results. ```<alpha>``` can be any positive non-zero float, though again, useful results will only be obtained for values in a certain interval. In the report, this was run for 2 cases: ```'original' 6 10``` and ```'custom' 7 7```.

----
### Bursting Neuron Case

In this notebook, the propagation of an "action potential", a spike of activity, along an axon (the part of a neuron along which the signal travels) is studied using the Hodgkin-Huxley model which has governing equation:

$$ C_{M} \frac{\partial V}{\partial t} = \frac{a}{2r_{L}}\frac{\partial^{2}V}{\partial x^{2}} + \sum_{j} \mathbf{I}_{j} $$

where $C_{M}$ is the membrane capacitance, $r_{L}$ is the resistivity inside the cell, and $I_{j}$ are ionic currents for each ion (e.g. Na, Ca, K, SI,). Here, 10 currents are considered, and the entire axon is considered spatially uniform, which means the equation describing the time-evolution of the membrane voltage due to external stimulation $I_{stim}$ is now:

$$ C_{M} \dot{V} = - \sum_{j} \mathbf{I}_{j} + I_{stim} $$

This notebook was written using:
- the already written ```Boundary_Layer.ipynb``` Notebook, and hence some of Callaham et al's written code,
- some of the custom alternative code written in that same Notebook,
- along with only the information in the paper and supplementary information document to try and reproduce the results shown for this case study in the paper.

For this case, the method is unique and the hyperparameters in the notebook are set so results match the paper's as closely as possible. Nonetheless there is still a script version one can run, still with the ability to choose the hyperparameters:

The script is run as follows:
```bash
$ cd Scripts/
$ python bursting_neuron.py <cluster number> <alpha>
```

```<cluster number>``` can be any non-zero integer and ```<alpha>``` can be any positive non-zero float, though again, results in the notebook and report were run with the parameters set as ```9 110``` to match the paper's results.

------
### Elasto-Inertial Turbulence Case

In this notebook, the Callaham et al. paper's method is used on simulation data of a polymer-laden flow, where the governing equation is much more complex, and some dominant balance regimes may be unknown still, unlike for well-studied cases in the paper (e.g. Turbulent Boundary Layer, Geostrophic Balance,...)

The governing equation is here given by:

$$ \partial_{t}\mathbf{u} + (\mathbf{u} \cdot \mathbf{\nabla})\mathbf{u} + \mathbf{\nabla}p = \frac{\beta}{Re} \Delta \mathbf{u} + \frac{1 - \beta}{Re} \mathbf{\nabla} \cdot \mathbf{T}(\mathbf{C}) $$

where

$ \mathbf{T}(\mathbf{C}) := \frac{1}{Wi}(f(\text{tr}\mathbf{C})\mathbf{C} - \mathbf{I}) $, and $ f(x) := (1 - \frac{x - 3}{L_{max}^{2}})^{-1} $.



Thus, $\mathbf{T}(\mathbf{C}) = \frac{1}{Wi} ((1 - \frac{(\text{tr}\mathbf{C}) - 3}{L_{max}^{2}})^{-1}\mathbf{C} - \mathbf{I}) $

This code was written using partly code from the original Callaham et al method, and partly using the alternative code which was written in the ```Boundary_Layer``` Notebook.

That dataset here contains simulation data for one of the attractors discussed in the [Beneitez et al. (2024)](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/multistability-of-elastoinertial-twodimensional-channel-flow/D63B7EDB638451A6FC2FBBFDA85E1BBD) paper. The attractor in question is the Chaotic Arrowhead Regime (CAR), which is characterised by a weak arrowhead structure near the centre of the channel. With EIT, it shares being maintained through near-wall mechanisms of energy transfers.

For now, only the x-component of the governing equation is studied:
$$ \partial_{t}u + u u_{x} + v u_{y} + p_{x} = \frac{\beta}{Re} (u_{xx} + u_{yy}) + \frac{1 - \beta}{Re} ( (\frac{1}{Wi} ((1 - \frac{(\text{tr}\mathbf{C}) - 3}{L_{max}^{2}})^{-1} C_{xx} - 1))_{x} + (\frac{1}{Wi} ((1 - \frac{(\text{tr}\mathbf{C}) - 3}{L_{max}^{2}})^{-1} C_{xy}))_{y} ) $$

Because this is novel data for which the true dominant balance regimes are not known, there are many valid results that can be obtained with multiple hyperparameters values. Thus the notebook covers the selection of hyperparameters and the obtaining of the dominant balance models for those values.

Because the notebook can be split into 2 parts, 2 python files were written for this case study. One to cover the selection of hyperparameters, and the other which simply goes through the Callaham et al. method for a chosen set of hyperparameters. They can be run as follows:

```bash
$ cd Scripts/
$ python EIT_param.py
$ python EIT.py <cluster number> <alpha>
```

Once again, the parameters can be any number subject to type and non-zero conditions. For the results presented in the report, the parameters chosen were 8 clusters and an $\alpha$ vlaue of 1.5.

----
### Geostrophic Balance in Ocean Currents Case

In this notebook, the ocean surface currents in the Gulf of Mexico are studied, modeled by the 2D incompressible Navier-Stokes equations on a rotating sphere:

$$ u_{t} + (\mathbf{u} \cdot \nabla)u + fv = -\frac{1}{\rho} p_{x} $$
$$ v_{t} + (\mathbf{u} \cdot \nabla)v - fu = -\frac{1}{\rho} p_{y} $$

where $\rho$ is the density, x and y are the zonal and meridional coordinates respectively. $f$, the Coriolis parameter is given in terms of the Earth's rotation rate $\Omega$ and the latitude $\phi$.

This code was written based on the paper [[1]](https://www.nature.com/articles/s41467-021-21331-z) and the supplementary information [[2]](https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-021-21331-z/MediaObjects/41467_2021_21331_MOESM1_ESM.pdf) provided for processing the data, and using some of the alternative code written in the Boundary Layer Notebook, though still using the Sklearn GMM and SPCA functions.

As usual, a script version can be run as follows:
```bash
$ cd Scripts/
$ python geostrophic_balance.py <cluster number> <alpha>
```

Same conditions apply. Values chosen in the notebook and report were a cluster number of 6, and an alpha value of 40.


------
### K-Means Turbulent Boundary Layer Case

As part of testing the method's robustness, other clusteirng algorithms were tested on the same turbulent boundary layer case. In this notebook, a standard K-means clustering algorithm is used instead of the GMM. The hyperparameters stay the same and again a script version was written.

And it can be run as follows:
```bash
$ cd Scripts/
$ python k_means_bl.py <cluster number> <alpha>
```

Same conditions apply. Values chosen in the notebook and report were a cluster number of 14, and an alpha value of 8.


------
### Optical Pulse Case

In this notebook, the case of generalized nonlinear Schrödinger equations is studied, in the context of a ultra-short pulse of light, typically occuring in optical fibers.

The governing equation was derived starting from Maxwell's wave equation in 1D, and is a PDE known as a generalized nonlinear Schrödinger equation (GNLSE):

$$ \frac{\partial u}{\partial x} - \sum_{k=2}^{\infty} \alpha_k \frac{\partial^k u}{\partial t^k} = \left(i - \frac{\partial}{\partial t}\right) u \int_{-\infty}^{\infty} r(t') | u(t') |^2 \text{d} t' \\
r(t) = a \delta(t) + b \exp (ct) \sin (dt) \Theta(t) $$

This describes the variation of the complex envelope $u(x, t)$ of the pulse. This equation is the nondimenionalized version (done with soliton scalings). The delta-function component of the RHS integral is called the cubic Kerr nonlinearity. And the r(t) function is called the Raman Kernel.

The ($\alpha_{k}, a, b, c, d$) describe the polarization response and were determined empirically.

This code was written using some code from the ```Boundary_Layer``` Notebook, which used some original Callaham et al. turbulent boundary layer case code and custom written alternative code. With that code and the information in the paper and its supplementary information, the aim here was to try and reproduce the results from the paper.

A script version was written and it can be run as follows:
```bash
$ cd Scripts/
$ python optical_pulse.py <cluster number> <alpha>
```

Same conditions apply. Values chosen in the notebook and report were a cluster number of 6, and an alpha value of 10.


------
### Spectral Clustering Turbulent Boundary Layer Case

In this notebook, the robustness of the method chosen in the Callaham paper is tested by exploring spectral clustering for unsupervised dominant balance identification. Spectral clustering functions by constructing a graph network between points based on a defined condition, which means it does not rely on Eulcidean Distance [Spectral CLustering](https://en.wikipedia.org/wiki/Spectral_clustering). This means it better able to handle different sized and shaped clusters. However, it does not have the capacity to take in new data points after training, which brings a computational cost difficulty.

The code here relied on the code written in the ```Boundary_Layer``` Notebook which has some code from the original Callaham et al paper's repository as well as custom written alternative code.

Here, because the computational cost is high, the training size fraction is also a hyperparameter. Thus the script version is run as:
```bash
$ cd Scripts/
$ python optical_pulse.py <training set size fraction> <cluster number> <alpha>
```
The training set size parameter can be any non-zero float between 0 and 1, as it is a fraction. For the other 2 arguments, the same conditions apply. Values chosen in the notebook and report were a training set size of 0.01, cluster number of 6, and an alpha value of 10.


-------
### Stability Assessment of the Callaham et al. Method

In this notebook, the stability of the Callaham proposed method is tested, by comparing the results obtained when modifying the hyperparameters of the algorithm (cluster number, training set size, alpha value).

For each hyperparameter being tested, the other values are kept fixed. Because it is a fairly time-consuming test, the script version takes in as only argument which hyperparameter one wishes to test:

```bash
$ cd Scripts/
$ python optical_pulse.py <hyperparameter>
```

In this case it can be one of 3: ```'n_cluster'```, ```'train_frac'```, and ```'alpha'```.


--------
### Weighted K-Means Turbulent Boundary Layer Case

Finally, a weighted K-Means algorithm is tried, where weights are defined as a function of the points' distance from the origin.

The weighted K-Means model is trained on a subset of the data with weights applied. The weights are defined as a function of the point's distance from the origin, and using the tanh() function:

$w = 1 - (\text{tanh}^{2}(\frac{1}{2}|\vec{OX}|))$

where $|\vec{OX}|$ is the point-origin distance.

These weights are between 0 and 1, and give more importance to points closer to the origin (e.g. plot below). This is so that the K-Means algorithm is encouraged to cluster points near the origin into separate groups.

As usual the script version can be run by setting the 2 standard hyperparameters:
```bash
$ cd Scripts/
$ python weigthed_k_means_bl.py <cluster number> <alpha>
```

Same conditions apply. Values chosen in the notebook and report were a cluster number of 6, and an alpha value of 7.

## Containerisation

For permissions reasons, the ```Dockerfile``` is not set up to pull the repository directly as it builds the image. Therefore, one must first download this repository to their local machine and then are free to build the Docker image from the ```Dockerfile```.

To run the solver on a Docker container, one first has to build the image and run the container. This can be done as follows:

```bash
$ docker build -t project_24 .
$ docker run --rm -ti project_24
```

The ```project_24``` is not a strict instruction, it can be set to any other name the user may prefer.

If there is a need to get the plots back on the local machine, the second line above can be ran without the ```--rm``` and can also set the container name using ```--name=container_name``` (any valid name is fine). From there, run all the code as instructed below. Once all desired outputs and plots have been obtained. One can exit the container and then run:

```bash
$ docker cp docker cp container_name:/Project_24/Plots ./Plots
```

## Tools

The ```Tools/``` directory contains all the modules used in the scripts and notebooks. In there are files for the plotting functions, the custom written GMM algorithm used in the alternative methods for the turbulent boundary layer case, and modular routines for the stability assessment scripts and notebook which were created for clarity.

## Plots

When running the code for the first time, a ```Plots/``` directory will be created and each case study will have it's own subdirectory. Most plots will follow a simple naming convention:

```<algorithm or method>_<what is plotted>_<hyperparams>.png```

## Report

Finally, there is a report directory which contains the final report PDF document, as well as a shorter executive summary of the project.
