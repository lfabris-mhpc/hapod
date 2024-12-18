from typing import Optional, Tuple

import numpy as np


def matrix_memory_footprint(shape: Tuple[int], dtype: np.dtype = np.float64):
    return np.prod(shape) * np.dtype(dtype).itemsize


def svd_memory_footprint(shape: Tuple[int], dtype: np.dtype = np.float64):
    return 2 * shape[-1] * (np.prod(shape[:-1]) + shape[-1]) * np.dtype(dtype).itemsize


def random_matrix(n_rows: int,
                  n_columns: int,
                  n_rank: int,
                  out: Optional[np.ndarray] = None,
                  rand_gen: Optional[np.random.Generator] = None,
                  dtype: np.dtype = np.float64):
    if rand_gen is None:
        rand_gen = np.random.default_rng()

    return np.dot(rand_gen.random((n_rows, min(n_rank, n_rows, n_columns)), dtype=dtype),
                  rand_gen.random((min(n_rank, n_rows, n_columns), n_columns), dtype=dtype), out)
