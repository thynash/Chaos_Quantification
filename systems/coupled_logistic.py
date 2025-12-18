import numpy as np

def coupled_logistic_map(r, eps, x0, y0, N, discard=100):
    """
    Symmetrically coupled logistic maps:
        x_{n+1} = r x_n (1 - x_n) + eps (y_n - x_n)
        y_{n+1} = r y_n (1 - y_n) + eps (x_n - y_n)

    Parameters
    ----------
    r : float
        Logistic parameter (chaotic regime)
    eps : float
        Coupling strength
    x0, y0 : float
        Initial conditions
    N : int
        Number of samples to return
    discard : int
        Number of transient iterations

    Returns
    -------
    traj : ndarray, shape (N, 2)
        Coupled system trajectory
    """

    """
    Diffusively coupled logistic maps:
        x_{n+1} = (1-eps) f(x_n) + eps f(y_n)
        y_{n+1} = (1-eps) f(y_n) + eps f(x_n)
    where f(x) = r x (1-x)

    This form preserves bounded dynamics.
    """

    x = np.empty(N + discard)
    y = np.empty(N + discard)

    x[0], y[0] = x0, y0

    for n in range(N + discard - 1):
        fx = r * x[n] * (1 - x[n])
        fy = r * y[n] * (1 - y[n])

        x[n+1] = (1 - eps) * fx + eps * fy
        y[n+1] = (1 - eps) * fy + eps * fx

    return np.column_stack((x[discard:], y[discard:]))

