import os
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import numpy as np


def matrix_memory_footprint(shape: Tuple[int], dtype: np.dtype = np.float64):
    return np.prod(shape) * np.dtype(dtype).itemsize


def svd_memory_footprint(shape: Tuple[int], dtype: np.dtype = np.float64):
    return 2 * shape[-1] * (np.prod(shape[:-1]) + shape[-1]) * np.dtype(dtype).itemsize


def random_matrix(n_rows: int,
                  n_columns: int,
                  n_rank: int,
                  out: Optional[np.ndarray] = None,
                  rand_gen: Optional[np.random.Generator] = None,
                  dtype: np.dtype = np.float64):
    if rand_gen is None:
        rand_gen = np.random.default_rng()

    return np.dot(rand_gen.random((n_rows, min(n_rank, n_rows, n_columns)), dtype=dtype),
                  rand_gen.random((min(n_rank, n_rows, n_columns), n_columns), dtype=dtype), out)


def is_file_in_dir(fname: str, dirname: str):
    fpath = os.path.abspath(fname)
    dpath = os.path.abspath(dirname)

    return os.path.commonpath([fpath, dpath]) == dpath


class TensorLoader(ABC):
    @abstractmethod
    def load(self, source: Union[np.ndarray, str]) -> np.ndarray:
        pass


class NumpyLoader(TensorLoader):
    def __init__(self, npz_fieldname: Optional[str] = None):
        self.npz_fieldname = npz_fieldname

    def load(self, source):
        if isinstance(source, np.ndarray):
            return source

        if isinstance(source, str):
            if not os.path.isfile(source):
                raise FileNotFoundError(f"File not found: {source}")

            if source.endswith(".npz"):
                loaded = np.load(source)
                if self.npz_fieldname is None or self.npz_fieldname not in loaded:
                    raise ValueError(f"Field {self.npz_fieldname} not found in the .npz file.")
                return loaded[self.npz_fieldname]

            return np.load(source)

        raise TypeError("Source must be either a string (file path) or a numpy.ndarray.")


#TODO: OpenFOAMLoader


def make_chunks(
    sources: List[Union[str, np.ndarray]],
    output_dir: str,
    n_chunks: Optional[int] = None,
    n_chunk_max_cols: Optional[int] = None,
    loader: Optional[TensorLoader] = None,
) -> List[str]:
    #compute actual n_chunks and/or chunk_cols
    n_sources = len(sources)

    if n_chunks is None and n_chunk_max_cols is None:
        raise ValueError(
            "invalid specification for chunking: either n_chunks or n_chunk_max_cols must be given"
        )

    if n_chunks is None:
        n_chunks = n_sources // n_chunk_max_cols
    elif n_chunk_max_cols is None:
        n_chunk_max_cols = int(np.ceil(n_sources / n_chunks))
    else:
        raise ValueError("exactly one of n_chunks and n_chunk_max_cols must be specified")

    os.makedirs(output_dir, exist_ok=True)

    if loader is None:
        loader = NumpyLoader("")

    chunk_fnames = []
    i_source = 0
    for i_chunk in range(n_chunks):
        chunk_size = n_sources // n_chunks
        if i_chunk < n_sources % n_chunks:
            chunk_size += 1

        chunk = []
        for source in sources[i_source:i_source + chunk_size]:
            chunk.append(loader.load(source).reshape(-1, 1))

        chunk = np.concatenate(tuple(chunk), axis=1)

        chunk_fname = os.path.join(output_dir, f"chunk_{i_chunk:04d}.npy")
        np.save(chunk_fname, chunk)
        chunk_fnames.append(chunk_fname)

        del chunk

        i_source += chunk_size

    return chunk_fnames
