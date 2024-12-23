import numpy as np

import hapod as hp

from test_base import *


def test_get_pod_eye_s():
    X = get_test_matrix_identity()

    _, s = hp.get_pod(X)

    assert np.allclose(s, 1)


def test_get_pod_full_rank():
    X, U_true, s_true = get_test_matrix_full_rank(return_Us=True)
    mask = ~np.isclose(s_true, 0)

    U, s = hp.get_pod(X)
    assert get_nonzero_close(s, s_true)

    print(U.shape)
    print(U_true.shape)

    ortho = hp.singular_vectors_orthogonality(U[:, mask], U_true[:, mask])
    assert np.allclose(ortho, 1)


def test_get_pod_half_rank():
    X, U_true, s_true = get_test_matrix_half_rank(return_Us=True)
    mask = ~np.isclose(s_true, 0)

    U, s = hp.get_pod(X)
    assert get_nonzero_close(s, s_true)

    ortho = hp.singular_vectors_orthogonality(U[:, mask], U_true[:, mask])
    assert np.allclose(ortho, 1)
