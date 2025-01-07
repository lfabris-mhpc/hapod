import os
import math
from typing import List, Optional, Tuple, Union

import numpy as np

from .serializer import MatrixSerializer, NumpySerializer


def get_matrix_from_svalues(
    n_rows: int,
    s: np.ndarray,
    dtype: np.dtype = np.float64,
    rng: Optional[np.random.Generator] = None,
    return_Us: bool = False,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()

    n_cols = len(s)
    n_tmp = min(n_rows, n_cols)
    U = np.zeros((n_rows, n_cols), dtype=dtype)
    U[:n_tmp], _ = np.linalg.qr(rng.random((n_tmp, n_cols), dtype=dtype))
    V, _ = np.linalg.qr(rng.random((n_cols, n_cols), dtype=dtype))
    X = U @ np.diag(s) @ V.T

    if return_Us:
        return X, U, s

    return X


def make_chunks(
    sources: List[Union[str, np.ndarray]],
    output_dir: str,
    n_chunks: Optional[int] = None,
    n_chunk_max_cols: Optional[int] = None,
    serializer: Optional[MatrixSerializer] = None,
    randomizer_rng: Optional[np.random.Generator] = None,
) -> List[str]:
    """
    Helper function to aggregate sources (interpreted as columns) into a list of chunked files.
    Loaded matrices are flattened before being concatenated in a chunk.
    Exactly one of n_chunks and n_chunk_max_cols must be specified.
    Returns the list of filenames created.

    Args:
        sources (List[Union[str, np.ndarray]]): list of columns, given as np.ndarray and/or source strings to be passed to the loader
        output_dir (str): output directory that will contain the files created
        n_chunks (Optional[int], optional): number of chunks to create. Defaults to None.
        n_chunk_max_cols (Optional[int], optional): maximum number of columns of a chunk. Defaults to None.
        loader (Optional[MatrixLoader], optional): loader instance to interpret a split when it is not a np.ndarray. If None, it uses NumpyLoader. Defaults to None.

    Returns:
        List[str]: list of filenames created
    """
    n_sources = len(sources)
    if randomizer_rng is not None:
        sources = list(sources)
        randomizer_rng.shuffle(sources)

    if n_chunks is None and n_chunk_max_cols is None:
        raise ValueError("exactly one of n_chunks and n_chunk_max_cols must be specified")

    if n_chunks is None:
        n_chunks = n_sources // n_chunk_max_cols
    elif n_chunk_max_cols is None:
        n_chunk_max_cols = int(np.ceil(n_sources / n_chunks))
    else:
        raise ValueError("exactly one of n_chunks and n_chunk_max_cols must be specified")

    os.makedirs(output_dir, exist_ok=True)

    if serializer is None:
        serializer = NumpySerializer("")

    snapshot_shape, snapshot_dtype = serializer.peek(sources[0])

    chunk_fnames = []
    i_source = 0
    for i_chunk in range(n_chunks):
        chunk_size = n_sources // n_chunks
        if i_chunk < n_sources % n_chunks:
            chunk_size += 1

        chunk = np.empty((np.prod(snapshot_shape), chunk_size), dtype=snapshot_dtype)
        for j, source in enumerate(sources[i_source:i_source + chunk_size]):
            chunk[:, j] = serializer.load(source).flatten()

        chunk_fname = os.path.join(output_dir, f"chunk_{i_chunk:04d}")
        chunk_fname = serializer.store(chunk, chunk_fname)
        chunk_fnames.append(chunk_fname)

        del chunk

        i_source += chunk_size

    return chunk_fnames


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


def get_pod(X: np.ndarray, rank_max: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return the truncated modes matrix and singular values of X using numpy.linalg.svd

    :param X: a 2d array to be decomposed, assumed X.shape[0] >= X.shape[1]
    :type X: np.ndarray
    :param rank_max: maximum number of singular values, defaults to None
    :type rank_max: Optional[int], optional
    :return: U 2d matrix of modes, s array of singular values
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    U, s, _ = np.linalg.svd(X, full_matrices=False)
    if rank_max is None:
        rank_max = len(s)

    return U[:, :rank_max], s[:rank_max]


def singular_vectors_orthogonality(U1: np.ndarray, U2: np.ndarray) -> np.ndarray:
    if U1.shape != U2.shape:
        raise ValueError(f"array shapes must be equal {U1.shape} vs {U2.shape}")
    return np.max(np.abs(U1.T @ U2), axis=0)


def peek_chunks_aggregation(
    sources: List[Union[str, np.ndarray]],
    serializer: Optional[MatrixSerializer] = None,
) -> Tuple[int, int]:
    if serializer is None:
        serializer = NumpySerializer("")

    n_rows = []
    n_cols = []
    dtypes = []
    for source in sources:
        shape, dtype = serializer.peek(source)
        if len(shape) != 2:
            raise ValueError(f"chunks must be 2d found {shape}")

        if n_rows and shape[0] != n_rows[-1]:
            raise ValueError(
                f"chunks must have the same number of rows found {shape[0]} vs {n_rows[-1]}")

        if dtypes and dtype != dtypes[-1]:
            raise ValueError(f"chunks must be of the same dtype found {dtype} vs {dtypes[-1]}")

        n_rows.append(shape[0])
        n_cols.append(shape[1])
        dtypes.append(dtype)

    return (n_rows[0], sum(n_cols)), dtypes[0]


def randomized_pod(
    sources: List[Union[str, np.ndarray]],
    n_sources_samples: int,
    serializer: Optional[MatrixSerializer] = None,
    rng: Optional[np.random.Generator] = None,
):
    if serializer is None:
        serializer = NumpySerializer("")

    if rng is None:
        rng = np.random.default_rng()

    (n_rows, n_cols), dtype = peek_chunks_aggregation(sources, serializer)
    n_sources_samples = min(n_sources_samples, n_cols)
    random_samples = rng.choice(n_cols, n_sources_samples, replace=False)
    random_samples.sort()

    Z = np.empty((n_rows, n_sources_samples), dtype=dtype)

    chunks_sizes = [serializer.peek(source)[0][1] for source in sources]
    chunks_ubs = np.cumsum(chunks_sizes)

    chunk = None
    j_chunk_prev = None
    for i, i_sample in enumerate(random_samples):
        j_chunk = np.searchsorted(chunks_ubs, i_sample, side="right")
        if j_chunk != j_chunk_prev:
            chunk = serializer.load(sources[j_chunk])
            j_chunk_prev = j_chunk

        i_chunk_offset = i_sample
        if j_chunk:
            i_chunk_offset -= chunks_ubs[j_chunk - 1]

        Z[:, i] = np.atleast_2d(chunk)[:, i_chunk_offset]

    rng.shuffle(Z, axis=1)

    Q, _ = np.linalg.qr(Z)
    Y = np.empty((n_sources_samples, n_cols), dtype=dtype)
    i_start = 0
    for source in sources:
        (_, sample_cols), _ = serializer.peek(source)
        i_end = i_start + sample_cols
        Y[:, i_start:i_end] = Q.T @ serializer.load(source)
        i_start = i_end

    Uy, s, _ = np.linalg.svd(Y, full_matrices=False)
    U = Q @ Uy

    return U, s


def chunked_sV(
    sources: List[Union[str, np.ndarray]],
    serializer: Optional[MatrixSerializer] = None,
    eigenvtol: float = 1e-16,
):
    if serializer is None:
        serializer = NumpySerializer("")

    (_, S_cols), S_dtype = peek_chunks_aggregation(sources)
    chunks_cols = [serializer.peek(chunk_fname)[0][1] for chunk_fname in sources]

    StS = np.zeros((S_cols, S_cols), S_dtype)

    StS_j_start = 0
    for j, chunk_right_fname in enumerate(sources):
        chunk_right = serializer.load(chunk_right_fname)

        StS_j_end = StS_j_start + chunks_cols[j]

        StS_i_start = sum(chunks_cols[:j])
        for i, chunk_left_fname in enumerate(sources[j:], j):
            if i != j:
                chunk_left = serializer.load(chunk_left_fname).T
            else:
                chunk_left = chunk_right.T

            StS_i_end = min(StS_i_start + chunks_cols[i], S_cols)

            StS[StS_i_start:StS_i_end, StS_j_start:StS_j_end] = chunk_left @ chunk_right

            StS_i_start = StS_i_end
        StS_j_start = StS_j_end

    s, V = np.linalg.eigh(StS, "L")
    masking = s <= eigenvtol
    s[masking] = 0
    sorter = np.argsort(s)[::-1]
    s = np.sqrt(s[sorter])
    V = V[:, sorter]

    return s, V


def chunked_U(
    sources: List[Union[str, np.ndarray]],
    s: np.ndarray,
    V: np.ndarray,
    serializer: Optional[MatrixSerializer] = None,
):
    if serializer is None:
        serializer = NumpySerializer("")

    (S_rows, _), S_dtype = peek_chunks_aggregation(sources)
    chunks_cols = [serializer.peek(chunk_fname)[0][1] for chunk_fname in sources]

    U = np.zeros((S_rows, V.shape[1]), S_dtype)
    Vsminus = V.copy()
    for i in range(len(s)):
        if np.abs(s[i]) > 1e-8:
            Vsminus[:, i] /= s[i]
    V_i_start = 0
    for j, chunk_fname in enumerate(sources):
        chunk = serializer.load(chunk_fname)

        V_i_end = V_i_start + chunks_cols[j]

        U += chunk @ Vsminus[V_i_start:V_i_end]

        V_i_start = V_i_end

    return U
