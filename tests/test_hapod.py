import numpy as np

import hapod as hp

from test_utils import *


def test_hapod_eye_s_1():
    X = get_test_matrix_identity()

    _, s = hp.hapod(
        [X],
        chunk_rank_max=len(X) // 2,
        serializer=hp.InMemorySerializer(),
    )

    assert np.allclose(s, 1)


def test_hapod_eye_s_2():
    X = get_test_matrix_identity()

    _, s = hp.hapod(
        np.array_split(X, 2, axis=1),
        chunk_rank_max=len(X) // 2,
        serializer=hp.InMemorySerializer(),
    )

    assert np.allclose(s, 1)


def test_hapod_eye_s_3():
    X = get_test_matrix_identity()

    _, s = hp.hapod(
        np.array_split(X, 4, axis=1),
        chunk_rank_max=len(X) // 2,
        serializer=hp.InMemorySerializer(),
    )

    assert np.allclose(s, 1)


def test_hapod_full_rank_1():
    X, U_true, s_true = get_test_matrix_full_rank(return_Us=True)
    mask = ~np.isclose(s_true, 0)

    U, s = hp.hapod(
        [X],
        chunk_rank_max=len(X) // 2,
        serializer=hp.InMemorySerializer(),
    )
    assert np.allclose(s, s_true)

    print(U.shape)
    print(U_true.shape)

    ortho = hp.singular_vectors_orthogonality(U[:, mask], U_true[:, mask])
    assert np.allclose(ortho, 1)


def test_hapod_full_rank_2():
    X, U_true, s_true = get_test_matrix_full_rank(return_Us=True)
    mask = ~np.isclose(s_true, 0)

    U, s = hp.hapod(
        np.array_split(X, 2, axis=1),
        chunk_rank_max=len(X) // 2,
        serializer=hp.InMemorySerializer(),
    )
    assert np.allclose(s, s_true)

    print(U.shape)
    print(U_true.shape)

    ortho = hp.singular_vectors_orthogonality(U[:, mask], U_true[:, mask])
    assert np.allclose(ortho, 1)


def test_hapod_full_rank_3():
    X, U_true, s_true = get_test_matrix_full_rank(return_Us=True)
    mask = ~np.isclose(s_true, 0)

    U, s = hp.hapod(
        np.array_split(X, 4, axis=1),
        chunk_rank_max=len(X) // 2,
        serializer=hp.InMemorySerializer(),
    )
    assert np.allclose(s, s_true)

    print(U.shape)
    print(U_true.shape)

    ortho = hp.singular_vectors_orthogonality(U[:, mask], U_true[:, mask])
    assert np.allclose(ortho, 1)


def test_hapod_half_rank_1():
    X, U_true, s_true = get_test_matrix_half_rank(return_Us=True)
    mask = ~np.isclose(s_true, 0)

    U, s = hp.hapod(
        [X],
        chunk_rank_max=len(X) // 2,
        serializer=hp.InMemorySerializer(),
    )
    assert np.allclose(s, s_true)

    ortho = hp.singular_vectors_orthogonality(U[:, mask], U_true[:, mask])
    assert np.allclose(ortho, 1)


def test_hapod_half_rank_2():
    X, U_true, s_true = get_test_matrix_half_rank(return_Us=True)
    mask = ~np.isclose(s_true, 0)

    U, s = hp.hapod(
        np.array_split(X, 2, axis=1),
        chunk_rank_max=len(X) // 2,
        serializer=hp.InMemorySerializer(),
    )
    assert np.allclose(s, s_true)

    ortho = hp.singular_vectors_orthogonality(U[:, mask], U_true[:, mask])
    assert np.allclose(ortho, 1)


def test_hapod_half_rank_3():
    X, U_true, s_true = get_test_matrix_half_rank(return_Us=True)
    mask = ~np.isclose(s_true, 0)

    U, s = hp.hapod(
        np.array_split(X, 4, axis=1),
        chunk_rank_max=len(X) // 2,
        serializer=hp.InMemorySerializer(),
    )
    assert np.allclose(s, s_true)

    ortho = hp.singular_vectors_orthogonality(U[:, mask], U_true[:, mask])
    assert np.allclose(ortho, 1)
