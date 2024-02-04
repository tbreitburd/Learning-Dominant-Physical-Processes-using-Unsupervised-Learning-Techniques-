import numpy as np
from scipy.integrate import odeint


eta_inf = 200
# Arbitrary "infinite" upper limit for domain
d_eta = 0.01
# Step size
eta = np.arange(0, eta_inf, d_eta)


def blasius_rhs(f):
    """RHS of Blasius equation recast as first order nonlinear ODE
    f[0] = f
    f[1] = f'
    f[2] = f''
    """
    return np.array([f[1], f[2], -f[0] * f[2] / 2])


def bc_fn(f0):
    """Solve with unknown initial condition as guess and evaluate at upper boundary"""

    global eta

    f = odeint(lambda f, t: blasius_rhs(f), f0, eta)
    # return discrepancy between upper boundary and desired f[2] = 1
    return [f0[0], f0[1], f[-1, 1] - 1]
