from typing import Tuple

import numpy as np

def matrix_memory_footprint(shape: Tuple[int], dtype: np.dtype=np.float64):
    return np.prod(shape) * np.dtype(dtype).itemsize

def svd_memory_footprint(shape: Tuple[int], dtype: np.dtype=np.float64):
    return 2 * shape[-1] * (np.prod(shape[:-1]) + shape[-1]) * np.dtype(dtype).itemsize

def random_matrix(n_rows: int, n_columns: int, n_rank: int):
    return np.random.randn(n_rows, min(n_rank, n_rows, n_columns)) @ np.random.randn(min(n_rank, n_rows, n_columns), n_columns)