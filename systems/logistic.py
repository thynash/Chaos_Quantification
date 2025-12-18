import numpy as np

def logistic_map(r, x0, N, discard=100):
    """
    Logistic map:
        x_{n+1} = r x_n (1 - x_n)

    Parameters
    ----------
    r : float
        Control parameter
    x0 : float
        Initial condition in (0,1)
    N : int
        Number of samples to return
    discard : int
        Number of transient iterations

    Returns
    -------
    x : ndarray, shape (N,)
        Time series
    """
    x = np.empty(N + discard)
    x[0] = x0

    for n in range(N + discard - 1):
        x[n+1] = r * x[n] * (1 - x[n])

    return x[discard:]

