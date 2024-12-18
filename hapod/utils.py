import os
import platform
import subprocess
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union
import zipfile

import numpy as np


def ram_size() -> int:
    """
    Try to retrieve the amount of RAM available to the current machine

    Returns:
        int: RAM in bytes
    """
    system = platform.system()

    if system == "Linux":
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    mem_total_kb = int(line.split()[1])
                    return mem_total_kb * 2**10

    elif system == "Darwin":
        # macOS
        result = subprocess.run(["sysctl", "hw.memsize"], stdout=subprocess.PIPE)
        mem_size_bytes = int(result.stdout.decode().split(":")[1].strip())
        return mem_size_bytes

    elif system == "Windows":
        result = subprocess.run(["wmic", "computersystem", "get", "TotalPhysicalMemory"],
                                stdout=subprocess.PIPE)
        mem_size_bytes = int(result.stdout.decode().split("\n")[1].strip())
        return mem_size_bytes

    else:
        raise OSError("Unsupported OS")


def matrix_memory_footprint(shape: Tuple[int], dtype: np.dtype = np.float64) -> int:
    """
    Computes the memory taken by a matrix with given shape and dtype, in bytes

    Args:
        shape (Tuple[int]): shape of the matrix
        dtype (np.dtype, optional): dtype of the matrix. Defaults to np.float64.

    Returns:
        int: bytes used by the given matrix
    """
    return np.prod(shape) * np.dtype(dtype).itemsize


def svd_memory_footprint(shape: Tuple[int], dtype: np.dtype = np.float64) -> int:
    """
    Computes the memory required by the svd of a matrix with given shape and dtype, in bytes

    Args:
        shape (Tuple[int]): shape of the matrix
        dtype (np.dtype, optional): dtype of the matrix. Defaults to np.float64.

    Returns:
        int: bytes used by the SVD of the given matrix
    """
    return 2 * 2 * shape[-1] * (np.prod(shape[:-1]) + shape[-1]) * np.dtype(dtype).itemsize


def random_matrix(n_rows: int,
                  n_columns: int,
                  n_rank: int,
                  out: Optional[np.ndarray] = None,
                  rand_gen: Optional[np.random.Generator] = None,
                  dtype: np.dtype = np.float64) -> np.ndarray:
    """
    Generates a random matrix with the required shape and rank

    Args:
        n_rows (int): number of rows in the output
        n_columns (int): number of columns in the output
        n_rank (int): rank of the resulting matrix
        out (Optional[np.ndarray], optional): Output argument. This must have the exact kind that would be returned if it was not used. In particular, it must have the right type, must be C-contiguous, and its dtype must be dtype. Defaults to None.
        rand_gen (Optional[np.random.Generator], optional): random generator to use. If None, uses np.random.default_rng(). Defaults to None.
        dtype (np.dtype, optional): dtype of the matrix. Defaults to np.float64.

    Returns:
        np.ndarray: output matrix
    """
    if rand_gen is None:
        rand_gen = np.random.default_rng()

    return np.dot(rand_gen.random((n_rows, min(n_rank, n_rows, n_columns)), dtype=dtype),
                  rand_gen.random((min(n_rank, n_rows, n_columns), n_columns), dtype=dtype), out)


class MatrixLoader(ABC):
    """
    Abstract base class for loading matrices during hapod
    """
    @abstractmethod
    def load(self, source: Union[np.ndarray, str]) -> np.ndarray:
        """
        Loads a np.ndarray from the given source

        Args:
            source (Union[np.ndarray, str]): either a np.ndarray, or a string representation of the source

        Returns:
            np.ndarray: the loaded np.ndarray
        """
        pass


class NumpyLoader(MatrixLoader):
    """
    MatrixLoader specialization to handle numpy .npy and .npz files
    """
    def __init__(self, npz_fieldname: Optional[str] = None):
        """
        Initialization

        Args:
            npz_fieldname (Optional[str], optional): the name of the array to be loaded when handling .npz files. Defaults to None.
        """
        self.npz_fieldname = npz_fieldname

    def peek(self, source):
        if isinstance(source, np.ndarray):
            return source.shape, source.dtype

        if isinstance(source, str):
            if not os.path.isfile(source):
                raise FileNotFoundError(f"File not found: {source}")

            if source.endswith(".npz"):
                with zipfile.ZipFile(source, "r") as archive:
                    if self.npz_fieldname is None or self.npz_fieldname not in archive.namelist():
                        raise ValueError(f"Field {self.npz_fieldname} not found in the .npz file.")

                    with archive.open(self.npz_fieldname) as fin:
                        magic = np.lib.format.read_magic(fin)
                        if magic[0] != 1:
                            raise ValueError(
                                f"Unsupported .npy format version in {self.npz_fieldname}")

                        header = np.lib.format.read_array_header_1_0(fin)
                        return header[0], header[2]

            with open(source, "rb") as fin:
                magic = np.lib.format.read_magic(fin)
                if magic[0] != 1:
                    raise ValueError("Unsupported .npy format version")

                header = np.lib.format.read_array_header_1_0(fin)
                return header[0], header[2]

        raise TypeError("Source must be either a string (file path) or a numpy.ndarray.")

    def load(self, source):
        if isinstance(source, np.ndarray):
            return source

        if isinstance(source, str):
            if not os.path.isfile(source):
                raise FileNotFoundError(f"File not found: {source}")

            if source.endswith(".npz"):
                loaded = np.load(source, mmap_mode="r")
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
    loader: Optional[MatrixLoader] = None,
) -> List[str]:
    """
    Helper function to aggregate sources (interpreted as columns) into a list of chunked files.
    Loaded matrices are flattened before being concatenated in a chunk.
    Exactly one of n_chunks and n_chunk_max_cols must be specified.
    Returns the list of filenames created.

    Args:
        sources (List[Union[str, np.ndarray]]): list of columns, given as np.ndarray and/or source strings to be passed to the loader
        output_dir (str): output directory that will contain the files created
        n_chunks (Optional[int], optional): number of chunks to create. Defaults to None.
        n_chunk_max_cols (Optional[int], optional): maximum number of columns of a chunk. Defaults to None.
        loader (Optional[MatrixLoader], optional): loader instance to interpret a split when it is not a np.ndarray. If None, it uses NumpyLoader. Defaults to None.

    Returns:
        List[str]: list of filenames created
    """
    n_sources = len(sources)

    if n_chunks is None and n_chunk_max_cols is None:
        raise ValueError("exactly one of n_chunks and n_chunk_max_cols must be specified")

    if n_chunks is None:
        n_chunks = n_sources // n_chunk_max_cols
    elif n_chunk_max_cols is None:
        n_chunk_max_cols = int(np.ceil(n_sources / n_chunks))
    else:
        raise ValueError("exactly one of n_chunks and n_chunk_max_cols must be specified")

    os.makedirs(output_dir, exist_ok=True)

    if loader is None:
        loader = NumpyLoader("")

    snapshot_shape, snapshot_dtype = loader.peek(sources[0])

    chunk_fnames = []
    i_source = 0
    for i_chunk in range(n_chunks):
        chunk_size = n_sources // n_chunks
        if i_chunk < n_sources % n_chunks:
            chunk_size += 1

        chunk = np.empty((np.prod(snapshot_shape), chunk_size), dtype=snapshot_dtype)
        for j, source in enumerate(sources[i_source:i_source + chunk_size]):
            chunk[:, j] = loader.load(source).flatten()

        chunk_fname = os.path.join(output_dir, f"chunk_{i_chunk:04d}.npy")
        np.save(chunk_fname, chunk)
        chunk_fnames.append(chunk_fname)

        del chunk

        i_source += chunk_size

    return chunk_fnames
