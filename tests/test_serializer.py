import os
import tempfile

import numpy as np

import hapod as hp

from test_base import *


def test_npy():
    X = get_test_matrix_half_rank()

    with tempfile.TemporaryDirectory() as work_dir:
        basename = os.path.join(work_dir, "test_npy")

        serializer = hp.NumpySerializer("")
        fname = serializer.store(X, basename)
        assert fname.endswith(".npy")

        shape, dtype = serializer.peek(fname)

        assert shape == X.shape
        assert dtype == X.dtype

        X_loaded = serializer.load(fname)
        assert X_loaded.shape == X.shape
        assert X_loaded.dtype == X.dtype
        assert np.allclose(X_loaded, X)


def test_npz():
    X = get_test_matrix_half_rank()

    with tempfile.TemporaryDirectory() as work_dir:
        basename = os.path.join(work_dir, "test_npz")

        serializer = hp.NumpySerializer("asd")
        fname = serializer.store(X, basename)
        assert fname.endswith(".npz")

        shape, dtype = serializer.peek(fname)

        assert shape == X.shape
        assert dtype == X.dtype

        X_loaded = serializer.load(fname)
        assert X_loaded.shape == X.shape
        assert X_loaded.dtype == X.dtype
        assert np.allclose(X_loaded, X)
