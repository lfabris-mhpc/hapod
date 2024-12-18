from typing import Optional, Callable, Tuple, Union, List
import time

import numpy as np


def get_cumulative_energy_ratios(s: np.ndarray) -> np.ndarray:
    """
    Compute cumulative, relative energy of the given array
    of singular values

    :param s: array of singular values, sorted desc
    :type s: np.ndarray
    :return: an array of same size as s
    :rtype: np.ndarray
    """
    if not len(s):
        return np.array([])

    return np.cumsum(s**2) / np.sum(s**2)


def get_truncation_rank(s: np.ndarray,
                        rank_max: Optional[int] = None,
                        magnitude_ratio_max: Optional[float] = None,
                        res_energy_ratio_max: Optional[float] = None) -> int:
    """
    Compute the appropriate truncation rank, taken as the tightest
    of the specified thresholds

    :param s: array of singular values, sorted desc
    :type s: np.ndarray
    :param rank_max: maximum number of singular values, defaults to None
    :type rank_max: Optional[int], optional
    :param magnitude_ratio_max: discard singular values whose relative magnitude is lower than the given value, defaults to None
    :type magnitude_ratio_max: Optional[float], optional
    :param res_energy_ratio_max: discard singular values whose cumulative relative energy is lower than the given value, defaults to None
    :type res_energy_ratio_max: Optional[float], optional
    :return: the minimum truncation rank
    :rtype: int
    """
    rmax = len(s)
    if not rmax:
        raise ValueError("empty singular values array")

    if rank_max is not None:
        rmax = min(rmax, rank_max)
    if magnitude_ratio_max is not None:
        r = np.searchsorted(np.flip(s) / np.max(s), magnitude_ratio_max)
        rmax = min(rmax, len(s) - r)
    if res_energy_ratio_max is not None:
        e = get_cumulative_energy_ratios(s)
        r = np.searchsorted(np.flip(1 - e), res_energy_ratio_max)
        rmax = min(rmax, len(s) - r)

    return max(1, rmax)


def get_pod(X: np.ndarray,
            rank_max: Optional[int] = None,
            magnitude_ratio_max: Optional[float] = None,
            res_energy_ratio_max: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return the truncated modes matrix and singular values of A

    :param X: a 2d array to be decomposed, assumed A.shape[0] >= A.shape[1]
    :type X: np.ndarray
    :param rank_max: maximum number of singular values, defaults to None
    :type rank_max: Optional[int], optional
    :param magnitude_ratio_max: discard singular values whose relative magnitude is lower than the given value, defaults to None
    :type magnitude_ratio_max: Optional[float], optional
    :param res_energy_ratio_max: discard singular values whose cumulative relative energy is lower than the given value, defaults to None
    :type res_energy_ratio_max: Optional[float], optional
    :return: U 2d matrix of modes, s array of singular values
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    U, s, _ = np.linalg.svd(X, full_matrices=False)
    rmax = get_truncation_rank(s,
                               rank_max=rank_max,
                               magnitude_ratio_max=magnitude_ratio_max,
                               res_energy_ratio_max=res_energy_ratio_max)

    return U[:, :rmax], s[:rmax]


def get_column_batches(X: np.ndarray, batch_size: int, debug: bool = False) -> List[np.ndarray]:
    """
    Split 2d matrix X in a list of submatrices, each with a subset of X's columns
    of size at most batch

    :param X: the 2d matrix to be split
    :type X: np.ndarray
    :param batch: maximum number of columns of each submatrix
    :type batch_size: int
    :param debug: whether to print debug informations, defaults to False
    :type debug: bool, optional
    :return: a list of submatrices of X, each with at most batch columns
    :rtype: List[np.ndarray]
    """
    Xs = []
    n_batches = int(np.ceil(X.shape[-1] / batch_size))
    size = int(np.floor(X.shape[-1] / n_batches))
    rest = X.shape[-1] - n_batches * size
    if debug:
        print(f"len {X.shape[-1]} batch {batch_size}")
        print(f"n_batches {n_batches} rest {rest} size {size}")
    assert n_batches * size + rest == X.shape[-1]

    start = 0
    for i in range(int(np.ceil(X.shape[-1] / batch_size))):
        end = start + size
        if i < rest:
            end += 1
        Xs.append(X[:, start:end])
        start = end

    return Xs


def hapod(Xs: List[Union[np.ndarray, str]],
          rank_max: Optional[int] = None,
          magnitude_ratio_max: Optional[float] = None,
          res_energy_ratio_max: Optional[float] = None,
          pod_impl: Callable[[np.ndarray, Optional[int], Optional[float], Optional[float]],
                             Tuple[np.ndarray, np.ndarray]] = get_pod,
          debug: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a Hierarchical Approximate Proper Orthogonal Decomposition
    of the matrix split in Xs

    :param Xs: list of 2d matrices, or filenames to be loaded, that contain the columns of the matrix to be decomposed
    :type Xs: List[Union[np.ndarray, str]]
    :param rank_max: maximum number of singular values, defaults to None
    :type rank_max: Optional[int], optional
    :param magnitude_ratio_max: discard singular values whose relative magnitude is lower than the given value, defaults to None
    :type magnitude_ratio_max: Optional[float], optional
    :param res_energy_ratio_max: discard singular values whose cumulative relative energy is lower than the given value, defaults to None
    :type res_energy_ratio_max: Optional[float], optional
    :param svd_impl: implementation of truncated SVD, defaults to get_truncated_svd
    :type svd_impl: Callable[[np.ndarray, Optional[int], Optional[float], Optional[float] ], Tuple[np.ndarray, np.ndarray]], optional
    :param debug: whether to print debug informations, defaults to False
    :type debug: bool, optional
    :return: U 2d matrix of modes, s array of singular values
    :rtype: Tuple[np.ndarray, np.ndarray]
    """

    Xs_local = list(Xs)

    if res_energy_ratio_max is not None and len(Xs_local) > 1:
        if debug:
            print(f"target res_energy_ratio_max {res_energy_ratio_max}")
        h = np.floor(1 + np.log2(len(Xs_local)))
        res_energy_ratio_max = 1 - np.power(1 - res_energy_ratio_max, 1 / h)
        if debug:
            print(f"estimated height {h}")
            print(f"actual res_energy_ratio_max {res_energy_ratio_max}")

    if debug:
        print("Xs")
        for x in Xs_local:
            if isinstance(x, str):
                print(f"    {x}")
            else:
                print(f"    {x.shape}")

    while len(Xs_local) > 1:
        X1, X2 = Xs_local.pop(0), Xs_local.pop(0)
        if isinstance(X1, str):
            X1 = np.load(X1, mmap_mode="r")
        if isinstance(X2, str):
            X2 = np.load(X2, mmap_mode="r")

        elapsed_svd = -time.perf_counter()
        Uu, ss = pod_impl(np.hstack((X1, X2)),
                          rank_max=rank_max,
                          magnitude_ratio_max=magnitude_ratio_max,
                          res_energy_ratio_max=res_energy_ratio_max)
        elapsed_svd += time.perf_counter()
        if debug:
            print(f"U_svd.shape {Uu.shape}")
            print(f"elapsed_svd {elapsed_svd:.3f} s")

        if not Xs_local:
            return Uu, ss

        Xs_local.append(Uu * ss[np.newaxis, :])

        if debug:
            print("Xs")
            for x in Xs_local:
                if isinstance(x, str):
                    print(f"    {x}")
                else:
                    print(f"    {x.shape}")

    if len(Xs_local) == 1:
        X1 = Xs_local.pop()
        if isinstance(X1, str):
            X1 = np.load(X1, mmap_mode="r")

        Uu, ss = pod_impl(X1,
                          rank_max=rank_max,
                          magnitude_ratio_max=magnitude_ratio_max,
                          res_energy_ratio_max=res_energy_ratio_max)
        return Uu, ss

    raise ValueError("Xs is empty")
