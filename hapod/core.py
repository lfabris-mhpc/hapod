import os
import shutil
import tempfile
import time
from typing import Optional, Callable, Tuple, Union, List

import numpy as np

from hapod.serializer import MatrixSerializer, NumpySerializer
from hapod.utils import get_pod


def hapod(
    Xs: List[Union[np.ndarray, str]],
    chunk_rank_max: int,
    pod_impl: Callable[
        [np.ndarray, Optional[int]], Tuple[np.ndarray, np.ndarray]
    ] = get_pod,
    serializer: Optional[MatrixSerializer] = None,
    temp_work_dir: Optional[str] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
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
        chunk_rank_max (int): maximum number of singular values from a merge.
        pod_impl (Callable[ [np.ndarray, Optional[int]], Tuple[np.ndarray, np.ndarray] ], optional): kernel for the pod. Defaults to get_pod.
        serializer (Optional[MatrixSerializer], optional): serializer instance to interpret a
            chunk when it is not a np.ndarray. If None, it uses NumpySerializer. Defaults to None.
        temp_work_dir (Optional[str], optional): output directory where the temporary merged
            files are stored. If None, a temporary directory is created.
            All files created by hapod will be removed. Defaults to None.
        verbose (bool, optional): whether to print diagnostic messages. Defaults to False.

    Raises:
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_

    Returns:
        Tuple[np.ndarray, np.ndarray]: U 2d matrix of modes as columns, s array of singular values
    """

    if serializer is None:
        serializer = NumpySerializer("")

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

    def source_repr(x: Union[np.ndarray, str], serializer: MatrixSerializer) -> str:
        if isinstance(x, str):
            return f"{x} {serializer.peek(x)}"

        return str(serializer.peek(x))

    if verbose:
        print("Xs")
        for x in Xs_local:
            print(f"    {source_repr(x, serializer)}")

    work_dir = temp_work_dir
    if work_dir is None:
        work_dir = tempfile.mkdtemp()
    else:
        os.makedirs(work_dir, exist_ok=True)

    merged_sources = set()

    def cleanup():
        for f in merged_sources:
            try:
                os.remove(f)
            except:
                pass

        if work_dir is not temp_work_dir:
            # delete only if created here
            shutil.rmtree(work_dir, ignore_errors=True)

    try:
        while len(Xs_local) > 1:
            X1_source = Xs_local.pop(0)
            X2_source = Xs_local.pop(0)

            elapsed_merge = -time.perf_counter()
            X1_shape, X1_dtype = serializer.peek(X1_source)
            X2_shape, X2_dtype = serializer.peek(X2_source)

            if verbose:
                print(f"merging ")
                print(f"    {source_repr(X1_source, serializer)}")
                print(f"    {source_repr(X2_source, serializer)}")

            if len(X1_shape) != len(X2_shape):
                raise ValueError("incompatible shapes")
            for d in range(len(X1_shape) - 1):
                if X1_shape[d] != X2_shape[d]:
                    raise ValueError("incompatible shapes")
            if X1_dtype != X2_dtype:
                raise ValueError("different dtype")

            # better to allocate once and keep no reference
            X_shape = tuple(list(X1_shape[:-1]) + [X1_shape[-1] + X2_shape[-1]])
            X = np.empty(X_shape, X1_dtype)
            X[:, : X1_shape[-1]] = serializer.load(X1_source)
            X[:, X1_shape[-1] :] = serializer.load(X2_source)
            # X = np.concatenate((loader.load(X1_source), loader.load(X2_source)), axis=-1)
            elapsed_merge += time.perf_counter()
            if verbose:
                print(f"    took {elapsed_merge:.3f}")

            if not Xs_local:
                # last merge is not truncated
                chunk_rank_max = None

            if verbose:
                print("POD")
            elapsed_pod = -time.perf_counter()
            U, s = pod_impl(
                X,
                rank_max=chunk_rank_max,
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

            X_tilde = U * s[np.newaxis, :]
            U = None
            del U
            s = None
            del s
            merged_fname = os.path.join(work_dir, f"merged_{len(merged_sources)}")
            merged_fname = serializer.store(X_tilde, merged_fname)
            if isinstance(merged_fname, str):
                merged_sources.add(merged_fname)
            X_tilde = None
            del X_tilde

            if verbose:
                print(f"    stored new chunk {source_repr(merged_fname, serializer)}")

            try:
                if X1_source in merged_sources:
                    os.remove(X1_source)
            except:
                pass
            try:
                if X2_source in merged_sources:
                    os.remove(X2_source)
            except:
                pass

            Xs_local.append(merged_fname)

            if verbose:
                print("Xs")
                for x in Xs_local:
                    print(f"    {source_repr(x, serializer)}")

        if len(Xs_local) == 1:
            # the list had a single element
            X1_source = Xs_local.pop(0)
            X1 = serializer.load(X1_source)

            if verbose:
                print(f"last chunk")
                print(f"    {source_repr(X1_source, serializer)}")

            elapsed_pod = -time.perf_counter()
            # last merge is not truncated
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
