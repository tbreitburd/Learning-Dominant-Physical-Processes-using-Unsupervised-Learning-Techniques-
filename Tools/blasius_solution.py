"""!@file blasius_solution.py

@brief Module to solve the Blasius boundary layer equation.

@details This module provides a function to solve the Blasius boundary layer equation.
This code was written by Callaham et al. (2021) and was placed in this module for
convenience.

@author T.Breitburd and Callaham et al."""

import numpy as np
from scipy.integrate import odeint

# Define some global variables
# Arbitrary "infinite" upper limit for domain
eta_inf = 200
# Step size
d_eta = 0.01

# Define domain
eta = np.arange(0, eta_inf, d_eta)


def blasius_rhs(f):
    """!@brief Compute the RHS of Blasius equation recast as first order nonlinear ODE

    @param f: array of dependent variables [f, f', f'']
    """
    return np.array([f[1], f[2], -f[0] * f[2] / 2])


def bc_fn(f0):
    """!@brief Compute the boundary conditions for the Blasius equation

    @param f0: array of initial conditions [f, f', f'']"""
    # Get eta
    global eta

    # Integrate the ODE
    f = odeint(lambda f, t: blasius_rhs(f), f0, eta)

    # Return discrepancy between upper boundary and desired f[2] = 1
    return [f0[0], f0[1], f[-1, 1] - 1]
