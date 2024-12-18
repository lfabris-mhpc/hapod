import os
import shutil
import tempfile
import time
from typing import Optional, Callable, Tuple, Union, List

import numpy as np

from .utils import MatrixLoader, NumpyLoader


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
    Return the truncated modes matrix and singular values of X

    :param X: a 2d array to be decomposed, assumed X.shape[0] >= X.shape[1]
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


def hapod(Xs: List[Union[np.ndarray, str]],
          rank_max: Optional[int] = None,
          magnitude_ratio_max: Optional[float] = None,
          res_energy_ratio_max: Optional[float] = None,
          pod_impl: Callable[[np.ndarray, Optional[int], Optional[float], Optional[float]],
                             Tuple[np.ndarray, np.ndarray]] = get_pod,
          loader: Optional[MatrixLoader] = None,
          temp_work_dir: Optional[str] = None,
          skip_last_truncation: bool = False,
          verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes a Hierarchical Approximate Proper Orthogonal Decomposition of the matrix split in Xs.
    The splits are merged in pairs, decomposed, and the resulting approximation is truncated.
    The matrix is then stored on disk, and its filename inserted in a queue for further merging.
    The memory footprint is kept at the minimum according to the truncation criteria.

    Args:
        Xs (List[Union[np.ndarray, str]]): list of 2d matrices, and/or filenames to be loaded, 
            that contain the columns of the matrix to be decomposed
        rank_max (Optional[int], optional): maximum number of singular values. Defaults to None.
        magnitude_ratio_max (Optional[float], optional): discard singular values whose relative 
            magnitude is lower than the given value. Defaults to None.
        res_energy_ratio_max (Optional[float], optional): discard singular values whose 
            cumulative relative energy is lower than the given value. Defaults to None.
        pod_impl (Callable[[np.ndarray, Optional[int], Optional[float], Optional[float]], 
            Tuple[np.ndarray, np.ndarray]], optional): implementation of POD taking a matrix 
                and the same truncation criteria. Defaults to get_pod.
        loader (Optional[TensorLoader], optional): loader instance to interpret a split when 
            it is not a np.ndarray. If None, it uses NumpyLoader. Defaults to None.
        temp_work_dir (Optional[str], optional): output directory where the temporary merged 
            files are stored. If None, a temporary directory is created. 
            All files created by hapod will be removed. Defaults to None.
        skip_last_truncation (bool, optional): whether to skip truncation of the bases and singular 
            values on the last merge. Defaults to False.
        verbose (bool, optional): whether to print diagnostic messages. Defaults to False.

    Raises:
        ValueError: ValueError if Xs is empty

    Returns:
        Tuple[np.ndarray, np.ndarray]: U 2d matrix of modes as columns, s array of singular values
    """

    if loader is None:
        loader = NumpyLoader("")

    if not Xs:
        raise ValueError("list of chunks is empty")

    Xs_local = list(Xs)

    if res_energy_ratio_max is not None and len(Xs_local) > 1:
        if verbose:
            print(f"target res_energy_ratio_max {res_energy_ratio_max}")
        h = np.floor(1 + np.log2(len(Xs_local)))
        res_energy_ratio_max = 1 - np.power(1 - res_energy_ratio_max, 1 / h)
        if verbose:
            print(f"estimated height {h} of the merge tree")
            print(f"actual res_energy_ratio_max {res_energy_ratio_max}")

    def source_repr(x: Union[np.ndarray, str]) -> str:
        if isinstance(x, np.ndarray):
            return str(x.shape)

        return x

    if verbose:
        print("Xs")
        for x in Xs_local:
            print(f"    {source_repr(x)}")

    work_dir = temp_work_dir
    if work_dir is None:
        work_dir = tempfile.mkdtemp()
    else:
        os.makedirs(work_dir, exist_ok=True)

    merged_fnames = set()

    def cleanup():
        for f in merged_fnames:
            if os.path.exists(f):
                os.remove(f)

        if work_dir is not temp_work_dir:
            shutil.rmtree(work_dir, ignore_errors=True)

    try:
        while len(Xs_local) > 1:
            X1_source = Xs_local.pop(0)
            X2_source = Xs_local.pop(0)

            X1 = loader.load(X1_source)
            X2 = loader.load(X2_source)
            if verbose:
                print(f"merging ")
                print(f"    {source_repr(X1_source)} {X1.shape}")
                print(f"    {source_repr(X2_source)} {X2.shape}")

            elapsed_pod = -time.perf_counter()
            X = np.concatenate((X1, X2), axis=1)
            del X1
            del X2

            if not Xs_local and skip_last_truncation:
                rank_max = None
                magnitude_ratio_max = None
                res_energy_ratio_max = None

                if verbose:
                    print(f"    skip last truncation")

            Uu, ss = pod_impl(X,
                              rank_max=rank_max,
                              magnitude_ratio_max=magnitude_ratio_max,
                              res_energy_ratio_max=res_energy_ratio_max)
            del X

            elapsed_pod += time.perf_counter()
            if verbose:
                print(f"    U.shape {Uu.shape}")
                print(f"    took {elapsed_pod:.3f}")

            if not Xs_local:
                cleanup()

                return Uu, ss

            merged_fname = os.path.join(work_dir, f"merged_{len(merged_fnames)}.npy")
            X_tilde = Uu * ss[np.newaxis, :]
            del Uu
            del ss
            np.save(merged_fname, X_tilde)
            merged_fnames.add(merged_fname)
            del X_tilde

            if verbose:
                print(f"    POD approximation in {merged_fname}")

            try:
                if X1_source in merged_fnames:
                    os.remove(X1_source)
            except:
                pass
            try:
                if X2_source in merged_fnames:
                    os.remove(X2_source)
            except:
                pass

            Xs_local.append(merged_fname)

            if verbose:
                print("Xs")
                for x in Xs_local:
                    print(f"    {source_repr(x)}")

        if len(Xs_local) == 1:
            X1_source = Xs_local.pop(0)
            X1 = loader.load(X1_source)

            if verbose:
                print(f"last chunk")
                print(f"    {source_repr(X1_source)} {X1.shape}")

            if skip_last_truncation:
                rank_max = None
                magnitude_ratio_max = None
                res_energy_ratio_max = None

                if verbose:
                    print(f"    skip last truncation")

            elapsed_pod = -time.perf_counter()
            Uu, ss = pod_impl(X1,
                              rank_max=rank_max,
                              magnitude_ratio_max=magnitude_ratio_max,
                              res_energy_ratio_max=res_energy_ratio_max)
            elapsed_pod += time.perf_counter()
            if verbose:
                print(f"    U.shape {Uu.shape}")
                print(f"    took {elapsed_pod:.3f}")

            del X1
    finally:
        cleanup()

    return Uu, ss
