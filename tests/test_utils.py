import os
import time

import numpy as np

from hapod import hapod as hp


def prepare_data(fname: str, batch_size: int = 100):
    """
    Load [fname].npy and perform svd (either exact, or randomized) storing the resulting U and s
    Also, split the matrix in column batches with at most batch_size columns and stores them in
    files [fname]_[i:02d].npy for later use

    :param fname: base name of the numpy array to load
    :type fname: str
    :param batch_size: _description_, defaults to 100
    :type batch_size: int, optional
    :return: U 2d matrix of modes, s array of singular values
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    U = None
    s = None

    if os.path.exists(f"{fname}_U.npy"):
        U = np.load(f"{fname}_U.npy")
    if os.path.exists(f"{fname}_s.npy"):
        s = np.load(f"{fname}_s.npy")

    if U is None or s is None:
        elapsed_svd = -time.perf_counter()
        X = np.load(f"{fname}.npy", mmap_mode="r")
        print(f"X.shape {X.shape}")

        if np.elements(X) <= 4 * 2**30:
            U, s, _ = np.linalg.svd(X, full_matrices=False)
        else:
            from sklearn.utils.extmath import randomized_svd
            U, s, _ = randomized_svd(
                X,
                n_components=int(X.shape[1] * .75),
                n_oversamples=100,
            )
        elapsed_svd += time.perf_counter()
        print(f"U.shape {U.shape}")
        print(f"elapsed_svd {elapsed_svd:.3f} s")

        np.save(f"{fname}_U.npy", U)
        np.save(f"{fname}_s.npy", s)

        Xs = hp.get_column_batches(X, batch=batch_size)
        for i, Xx in enumerate(Xs):
            np.save(f"{fname}_{i:02d}.npy", Xx)

        del Xs

    return U, s
