import numpy as np

def koopman_eigenvalues(K):
    return np.linalg.eigvals(K)


def spectral_gap(eigs):
    mags = np.sort(np.abs(eigs))[::-1]
    return mags[0] - mags[1]


def spectral_entropy(eigs):
    mags = np.abs(eigs)
    p = mags / np.sum(mags)
    return -np.sum(p * np.log(p + 1e-12))

