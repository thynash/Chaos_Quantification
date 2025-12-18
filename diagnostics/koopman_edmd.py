import numpy as np

# -------------------------------------------------
# Lifting functions
# -------------------------------------------------

def lift_state(x):
    """
    Nonlinear observable lifting.
    Works for 1D or multi-D x.
    """
    x = np.atleast_1d(x)
    features = [1.0]

    for xi in x:
        features.append(xi)
        features.append(xi**2)
        features.append(np.tanh(xi))

    return np.array(features)


# -------------------------------------------------
# EDMD Koopman approximation
# -------------------------------------------------

def koopman_edmd(X, Y):
    """
    Compute Koopman operator K from snapshot pairs
    using EDMD.

    X, Y: arrays of shape (N, d)
    """
    Psi_X = np.array([lift_state(x) for x in X])
    Psi_Y = np.array([lift_state(y) for y in Y])

    # Least-squares solution
    K = np.linalg.lstsq(Psi_X, Psi_Y, rcond=None)[0]
    return K

