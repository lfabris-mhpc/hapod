import os
import time

import numpy as np

import hapod as hp

from test_base import *


def test_get_memory_size():
    m = hp.get_memory_size()

    assert m > 1


def test_get_matrix_memory_footprint():
    shape = (1000, 500)
    m1 = hp.get_matrix_memory_footprint(shape, dtype=np.float64)
    m2 = hp.get_matrix_memory_footprint(shape, dtype=np.float32)

    assert m1 == 2 * m2


def test_get_svd_memory_footprint():
    shape = (1000, 500)
    m1 = hp.get_svd_memory_footprint(shape, dtype=np.float64)
    m2 = hp.get_svd_memory_footprint(shape, dtype=np.float32)

    assert m1 == 2 * m2


def test_get_max_svd_columns():
    shape = (1000, 500)
    m_limit = hp.get_memory_size() * 0.8
    m1 = hp.get_max_svd_columns(shape[0], m_limit, dtype=np.float64)
    m2 = hp.get_max_svd_columns(shape[0], m_limit, dtype=np.float32)

    assert abs(m2 - (2 * m1)) <= 2


def test_get_max_svd_square():
    m_limit = hp.get_memory_size() * 0.8
    m1 = hp.get_max_svd_square(m_limit, dtype=np.float64)
    m2 = hp.get_max_svd_square(m_limit, dtype=np.float32)

    assert np.isclose(m2 / m1, np.sqrt(2), atol=1e-4)
