import numpy as np
import pytest

import hapod as hp
from test_utils import *


def test_hapod_list_0():
    Xs = []
    with pytest.raises(ValueError):
        hp.hapod(Xs, rank_max=0)


def test_hapod_list_1(n_rows: int = 1000, n_cols: int = 100, rank_max: int = 100):
    X = get_trivial_matrix(n_rows, n_cols)
    Xs = [X]
    _, s = hp.hapod(Xs, rank_max)

    assert len(s) == rank_max


def test_hapod_list_3(n_rows: int = 1000, n_cols: int = 100, rank_max: int = 100):
    X = get_trivial_matrix(n_rows, n_cols)
    Xs = np.array_split(X, 3, axis=1)
    _, s = hp.hapod(Xs, rank_max)

    assert len(s) == rank_max
