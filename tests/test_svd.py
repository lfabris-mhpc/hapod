import numpy as np
import pytest

import hapod as hp
from test_utils import *


def test_hapod_list_0():
    Xs = []
    with pytest.raises(ValueError):
        hp.hapod(Xs)


def test_hapod_list_1(n: int = 1000, m: int = 100, rank_max: int = 100):
    X = get_trivial_matrix(n, m)
    Xs = [X]
    _, s = hp.hapod(Xs)

    assert len(s) == rank_max


def test_hapod_list_3(n: int = 1000, m: int = 100, rank_max: int = 100):
    X = get_trivial_matrix(n, m)
    Xs = hp.get_column_batches(X, batch_size=np.ceil(m / 3))
    _, s = hp.hapod(Xs)

    assert len(s) == rank_max


def test_hapod_magnitude_frac_max(n: int = 1000, m: int = 100, magnitude_frac_max: float = 1e-2):
    X = get_trivial_matrix(n, m)
    _, s = hp.hapod([X], magnitude_frac_max=magnitude_frac_max)

    assert len(s) == X.shape[1]


def test_hapod_rank_res_energy_frac_max(n: int = 1000,
                                        m: int = 100,
                                        res_energy_frac_max: float = 1e-2):
    X = get_trivial_matrix(n, m)
    _, s = hp.hapod([X], res_energy_frac_max=res_energy_frac_max)

    assert len(s) == np.floor(X.shape[1] * (1 - res_energy_frac_max))
