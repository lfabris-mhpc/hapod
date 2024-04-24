import numpy as np
from hapod import hapod as hp


def test_get_cumulative_energy_frac():
    s = np.flip(np.linspace(1, 11, 11))
    e = hp.get_cumulative_energy_frac(s)
    assert np.all(np.isclose(e, np.cumsum(s**2) / np.sum(s**2)))


def test_get_truncation_rank_rank_max():
    s = np.flip(np.linspace(1, 11, 11))
    rmax = hp.get_truncation_rank(s, rank_max=6)
    assert rmax == 6


def test_get_truncation_rank_magnitude_frac_max():
    s = np.flip(np.linspace(1, 11, 11))
    rmax = hp.get_truncation_rank(s, magnitude_frac_max=1e-2)
    assert rmax == 11


def test_get_truncation_rank_res_energy_frac_max():
    s = np.flip(np.linspace(1, 11, 11))
    rmax = hp.get_truncation_rank(s, res_energy_frac_max=1e-2)
    assert rmax == 8
