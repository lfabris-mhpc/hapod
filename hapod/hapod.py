import os
import shutil
import tempfile
import time
from typing import Optional, Callable, Tuple, Union, List

import numpy as np

from .utils import MatrixLoader, NumpyLoader, get_pod


def hapod(Xs: List[Union[np.ndarray, str]],
          rank_max: int,
          pod_impl: Callable[[np.ndarray, Optional[int]], Tuple[np.ndarray, np.ndarray]] = get_pod,
          loader: Optional[MatrixLoader] = None,
          temp_work_dir: Optional[str] = None,
          verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes a Hierarchical Approximate Proper Orthogonal Decomposition of the matrix obtained 
    by concatenation of the chunks in Xs.
    The chunks are loaded and merged in pairs, then decomposed using the pod_impl callable.
    If no more chunks remain to process, the U and s arrays are returned.
    Otherwise, U and s are truncated to rank_max and the array U * s is stored as new chunk.
    This approach trades off disk usage for the lack of memory.

    Args:
        Xs (List[Union[np.ndarray, str]]): list of 2d matrices, and/or filenames to be loaded, 
            that contain the columns of the matrix to be decomposed
        rank_max (int): maximum number of singular values from a merge.
        get_pod (Callable[[np.ndarray, Optional[int]], 
            Tuple[np.ndarray, np.ndarray]], optional): implementation of POD taking a matrix 
                and a truncation criterion. Defaults to get_pod.
        loader (Optional[TensorLoader], optional): loader instance to interpret a split when 
            it is not a np.ndarray. If None, it uses NumpyLoader. Defaults to None.
        temp_work_dir (Optional[str], optional): output directory where the temporary merged 
            files are stored. If None, a temporary directory is created. 
            All files created by hapod will be removed. Defaults to None.
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

    # let's keep a memory of the past
    # if res_energy_ratio_max is not None and len(Xs_local) > 1:
    #     if verbose:
    #         print(f"target res_energy_ratio_max {res_energy_ratio_max}")
    #     h = np.floor(1 + np.log2(len(Xs_local)))
    #     res_energy_ratio_max = 1 - np.power(1 - res_energy_ratio_max, 1 / h)
    #     if verbose:
    #         print(f"estimated height {h} of the merge tree")
    #         print(f"actual res_energy_ratio_max {res_energy_ratio_max}")

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

            elapsed_merge = -time.perf_counter()
            X1_shape, X1_dtype = loader.peek(X1_source)
            X2_shape, X2_dtype = loader.peek(X2_source)

            if verbose:
                print(f"merging ")
                print(f"    {source_repr(X1_source)} {X1_shape}")
                print(f"    {source_repr(X2_source)} {X2_shape}")

            if len(X1_shape) != len(X2_shape):
                raise ValueError("incompatible shapes")
            for d in range(len(X1_shape) - 1):
                if X1_shape[d] != X2_shape[d]:
                    raise ValueError("incompatible shapes")
            if X1_dtype != X2_dtype:
                raise ValueError("different dtype")

            X_shape = tuple(list(X1_shape[:-1]) + [X1_shape[-1] + X2_shape[-1]])
            X = np.empty(X_shape, X1_dtype)
            X[:, :X1_shape[-1]] = loader.load(X1_source)
            X[:, X1_shape[-1]:] = loader.load(X2_source)
            # X = np.concatenate((loader.load(X1_source), loader.load(X2_source)), axis=-1)
            elapsed_merge += time.perf_counter()
            if verbose:
                print(f"    took {elapsed_merge:.3f}")

            if not Xs_local:
                rank_max = None

            elapsed_pod = -time.perf_counter()
            U, s = pod_impl(
                X,
                rank_max=rank_max,
            )
            X = None
            del X

            elapsed_pod += time.perf_counter()
            if verbose:
                print(f"    U.shape {U.shape}")
                print(f"    took {elapsed_pod:.3f}")

            if not Xs_local:
                cleanup()

                return U, s

            merged_fname = os.path.join(work_dir, f"merged_{len(merged_fnames)}.npy")
            X_tilde = U * s[np.newaxis, :]
            U = None
            del U
            s = None
            del s
            np.save(merged_fname, X_tilde)
            merged_fnames.add(merged_fname)
            X_tilde = None
            del X_tilde

            if verbose:
                print(f"    stored new chunk {merged_fname}")

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

            elapsed_pod = -time.perf_counter()
            U, s = pod_impl(
                X1,
                rank_max=None,
            )
            elapsed_pod += time.perf_counter()
            if verbose:
                print(f"    U.shape {U.shape}")
                print(f"    took {elapsed_pod:.3f}")

            X1 = None
            del X1
    finally:
        cleanup()

    return U, s
