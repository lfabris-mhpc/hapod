import tempfile

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
    rank_max = np.sum(mask)

    U, s = hp.hapod(
        [X],
        chunk_rank_max=len(s_true),
        serializer=hp.InMemorySerializer(),
    )
    assert get_nonzero_close(s, s_true)

    print(U.shape)
    print(U_true.shape)

    ortho = hp.singular_vectors_orthogonality(U[:, mask], U_true[:, mask])
    assert np.allclose(ortho, 1)


def test_hapod_full_rank_2():
    X, U_true, s_true = get_test_matrix_full_rank(return_Us=True)
    mask = ~np.isclose(s_true, 0)
    rank_max = np.sum(mask)

    U, s = hp.hapod(
        np.array_split(X, 2, axis=1),
        chunk_rank_max=len(s_true),
        serializer=hp.InMemorySerializer(),
    )
    assert get_nonzero_close(s, s_true)

    print(U.shape)
    print(U_true.shape)

    ortho = hp.singular_vectors_orthogonality(U[:, mask], U_true[:, mask])
    assert np.allclose(ortho, 1)


def test_hapod_full_rank_3():
    X, U_true, s_true = get_test_matrix_full_rank(return_Us=True)
    mask = ~np.isclose(s_true, 0)
    rank_max = np.sum(mask)

    U, s = hp.hapod(
        np.array_split(X, 4, axis=1),
        chunk_rank_max=len(s_true),
        serializer=hp.InMemorySerializer(),
    )
    assert get_nonzero_close(s, s_true)

    print(U.shape)
    print(U_true.shape)

    ortho = hp.singular_vectors_orthogonality(U[:, mask], U_true[:, mask])
    assert np.allclose(ortho, 1)


def test_hapod_half_rank_1():
    X, U_true, s_true = get_test_matrix_half_rank(return_Us=True)
    mask = ~np.isclose(s_true, 0)
    rank_max = np.sum(mask)

    U, s = hp.hapod(
        [X],
        chunk_rank_max=len(s_true),
        serializer=hp.InMemorySerializer(),
    )
    assert get_nonzero_close(s, s_true)

    ortho = hp.singular_vectors_orthogonality(U[:, mask], U_true[:, mask])
    assert np.allclose(ortho, 1)


def test_hapod_half_rank_2():
    X, U_true, s_true = get_test_matrix_half_rank(return_Us=True)
    mask = ~np.isclose(s_true, 0)
    rank_max = np.sum(mask)

    U, s = hp.hapod(
        np.array_split(X, 2, axis=1),
        chunk_rank_max=len(s_true),
        serializer=hp.InMemorySerializer(),
    )
    assert get_nonzero_close(s, s_true)

    ortho = hp.singular_vectors_orthogonality(U[:, mask], U_true[:, mask])
    assert np.allclose(ortho, 1)


def test_hapod_half_rank_3():
    X, U_true, s_true = get_test_matrix_half_rank(return_Us=True)
    mask = ~np.isclose(s_true, 0)
    rank_max = np.sum(mask)

    U, s = hp.hapod(
        np.array_split(X, 4, axis=1),
        chunk_rank_max=len(s_true),
        serializer=hp.InMemorySerializer(),
    )
    assert get_nonzero_close(s, s_true)

    ortho = hp.singular_vectors_orthogonality(U[:, mask], U_true[:, mask])
    assert np.allclose(ortho, 1)


def test_hapod_file_1():
    X, U_true, s_true = get_test_matrix_half_rank(return_Us=True)
    mask = ~np.isclose(s_true, 0)
    rank_max = np.sum(mask)

    with tempfile.TemporaryDirectory() as work_dir:
        chunks_fnames = hp.make_chunks(
            np.array_split(X, len(s_true), axis=1),
            work_dir,
            n_chunks=4,
        )

        U, s = hp.hapod(
            chunks_fnames,
            chunk_rank_max=len(s_true),
        )
        assert get_nonzero_close(s, s_true)

        ortho = hp.singular_vectors_orthogonality(U[:, mask], U_true[:, mask])
        assert np.allclose(ortho, 1)


def test_hapod_file_2():
    X, U_true, s_true = get_test_matrix_half_rank(return_Us=True)
    mask = ~np.isclose(s_true, 0)
    rank_max = np.sum(mask)

    with tempfile.TemporaryDirectory() as work_dir:
        chunks_fnames = hp.make_chunks(
            np.array_split(X, len(s_true), axis=1),
            work_dir,
            n_chunks=hp.get_n_chunks_fulltree(X.shape[-1], rank_max),
        )

        U, s = hp.hapod(
            chunks_fnames,
            chunk_rank_max=len(s_true),
        )
        assert get_nonzero_close(s, s_true)

        ortho = hp.singular_vectors_orthogonality(U[:, mask], U_true[:, mask])
        assert np.allclose(ortho, 1)


def test_hapod_file_3():
    X, U_true, s_true = get_test_matrix_half_rank(return_Us=True)
    mask = ~np.isclose(s_true, 0)
    rank_max = np.sum(mask)

    with tempfile.TemporaryDirectory() as work_dir:
        chunks_fnames = hp.make_chunks(
            np.array_split(X, len(s_true), axis=1),
            work_dir,
            n_chunk_max_cols=rank_max,
        )

        U, s = hp.hapod(
            chunks_fnames,
            chunk_rank_max=len(s_true),
        )
        assert get_nonzero_close(s, s_true)

        ortho = hp.singular_vectors_orthogonality(U[:, mask], U_true[:, mask])
        assert np.allclose(ortho, 1)
