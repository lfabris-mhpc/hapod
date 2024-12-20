import numpy as np

import hapod as hp


def get_test_matrix_full_rank(n_rows: int = 1000,
                              n_cols: int = 500,
                              dtype: np.dtype = np.float64,
                              return_Us: bool = False):
    rng = np.random.default_rng(42)

    ranks = np.arange(0, n_cols)

    s_true = np.exp(-0.1 * ranks).astype(dtype)

    return hp.get_matrix_from_svalues(n_rows, s_true, dtype=dtype, rng=rng, return_Us=return_Us)


def get_test_matrix_half_rank(n_rows: int = 1000,
                              n_cols: int = 500,
                              dtype: np.dtype = np.float64,
                              return_Us: bool = False):
    rng = np.random.default_rng(42)

    ranks = np.arange(0, n_cols)

    s_true = np.exp(-0.1 * ranks).astype(dtype)
    s_true[n_cols // 2:] = 0

    return hp.get_matrix_from_svalues(n_rows, s_true, dtype=dtype, rng=rng, return_Us=return_Us)


def get_test_matrix_identity(
    n_rows: int = 1000,
    dtype: np.dtype = np.float64,
):
    rng = np.random.default_rng(42)

    s_true = np.ones(n_rows)

    return hp.get_matrix_from_svalues(n_rows, s_true, dtype=dtype, rng=rng)
