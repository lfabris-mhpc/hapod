import os
import platform
import subprocess
import zipfile
import math
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import numpy as np


def get_memory_size() -> int:
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


def get_matrix_memory_footprint(shape: Tuple[int], dtype: np.dtype = np.float64) -> int:
    """
    Computes the memory taken by a matrix with given shape and dtype, in bytes

    Args:
        shape (Tuple[int]): shape of the matrix
        dtype (np.dtype, optional): dtype of the matrix. Defaults to np.float64.

    Returns:
        int: bytes used by the given matrix
    """
    return np.prod(shape) * np.dtype(dtype).itemsize


def get_svd_memory_footprint(shape: Tuple[int], dtype: np.dtype = np.float64) -> int:
    """
    Computes the memory required by the svd of a matrix with given shape and dtype, in bytes

    Args:
        shape (Tuple[int]): shape of the matrix
        dtype (np.dtype, optional): dtype of the matrix. Defaults to np.float64.

    Returns:
        int: bytes used by the SVD of the given matrix
    """
    v_size = min(shape[-2], shape[-1])
    return 2.18 * (math.prod(shape) +
                   (math.prod(shape) // v_size) * v_size + v_size) * np.dtype(dtype).itemsize


def get_max_svd_columns(n_rows: int,
                        memory_limit: Optional[int] = None,
                        dtype: np.dtype = np.float64) -> int:
    if memory_limit is None:
        memory_limit = get_memory_size()
    itemsize = np.dtype(dtype).itemsize

    lb = 1
    ub = int(memory_limit) // (itemsize * n_rows)
    while (ub - lb) > 0:
        n_cols = (ub + lb + 1) // 2
        ram_req = get_svd_memory_footprint((n_rows, n_cols), dtype)

        if ram_req > memory_limit:
            ub = n_cols - 1
        elif ram_req < memory_limit:
            lb = n_cols
        else:
            return n_cols

    return lb


def get_max_svd_square(memory_limit: Optional[int] = None, dtype: np.dtype = np.float64) -> int:
    if memory_limit is None:
        memory_limit = get_memory_size()
    itemsize = np.dtype(dtype).itemsize

    lb = 1
    ub = int(memory_limit) // itemsize
    while (ub - lb) > 0:
        n_cols = (ub + lb + 1) // 2
        ram_req = get_svd_memory_footprint((n_cols, n_cols), dtype)

        # print(f"bounds {lb, ub}")
        # print(n_cols)
        if ram_req > memory_limit:
            ub = n_cols - 1
        elif ram_req < memory_limit:
            lb = n_cols
        else:
            return n_cols

    return lb


def get_n_chunks_fulltree(n_cols: int, n_chunk_max_cols: int) -> int:
    return 2**int(np.ceil(np.log2(n_cols / n_chunk_max_cols)))


def get_random_matrix(n_rows: int,
                      n_cols: int,
                      n_rank: int,
                      out: Optional[np.ndarray] = None,
                      rand_gen: Optional[np.random.Generator] = None,
                      dtype: np.dtype = np.float64) -> np.ndarray:
    """
    Generates a random matrix with the required shape and rank

    Args:
        n_rows (int): number of rows in the output
        n_cols (int): number of columns in the output
        n_rank (int): rank of the resulting matrix
        out (Optional[np.ndarray], optional): Output argument. This must have the exact kind that would be returned if it was not used. In particular, it must have the right type, must be C-contiguous, and its dtype must be dtype. Defaults to None.
        rand_gen (Optional[np.random.Generator], optional): random generator to use. If None, uses np.random.default_rng(). Defaults to None.
        dtype (np.dtype, optional): dtype of the matrix. Defaults to np.float64.

    Returns:
        np.ndarray: output matrix
    """
    if rand_gen is None:
        rand_gen = np.random.default_rng()

    return np.dot(rand_gen.random((n_rows, min(n_rank, n_rows, n_cols)), dtype=dtype),
                  rand_gen.random((min(n_rank, n_rows, n_cols), n_cols), dtype=dtype), out)


class MatrixSerializer(ABC):
    """
    Abstract base class for loading matrices during hapod
    """
    @abstractmethod
    def peek(self, source: Union[np.ndarray, str]) -> Tuple[Tuple[int], np.dtype]:
        """
        Retrieve the shape and dtype of the source array, 
        possibly without loading the entire file in memory

        Args:
            source (Union[np.ndarray, str]): either a np.ndarray, or a string representation of the source

        Returns:
            Tuple[Tuple[int], np.dtype]: shape and dtype of the source
        """
        pass

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

    @abstractmethod
    def store(self, X: np.ndarray, basename: str) -> Union[np.ndarray, str]:
        """
        Store the given array using the given basename if needed.
        Return the source identifier that will be used in the future to load back the array.

        Args:
            X (np.ndarray): the array to store
            basename (str): the basename to interpret and possibly modify

        Returns:
            Union[np.ndarray, str]: either the array, if kept in memory, or a string to be fed to the load method
        """
        pass


class InMemorySerializer(MatrixSerializer):
    def peek(self, source: Union[np.ndarray, str]) -> Tuple[Tuple[int], np.dtype]:
        if isinstance(source, np.ndarray):
            return source.shape, source.dtype

        raise TypeError("Source must be a numpy.ndarray.")

    def load(self, source: Union[np.ndarray, str]) -> np.ndarray:
        if isinstance(source, np.ndarray):
            return source

        raise TypeError("Source must be a numpy.ndarray.")

    def store(self, X: np.ndarray, basename: str) -> Union[np.ndarray, str]:
        return X


class NumpySerializer(MatrixSerializer):
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

    def peek(self, source: Union[np.ndarray, str]) -> Tuple[Tuple[int], np.dtype]:
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

    def load(self, source: Union[np.ndarray, str]) -> np.ndarray:
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

    def store(self, X: np.ndarray, basename: str) -> Union[np.ndarray, str]:
        fname = None
        if self.npz_fieldname:
            fname = basename + ".npz"
            np.savez_compressed(fname, {self.npz_fieldname: X})
        else:
            fname = basename + ".npy"
            np.save(fname, X)

        return fname


#TODO: OpenFOAMLoader


def make_chunks(
    sources: List[Union[str, np.ndarray]],
    output_dir: str,
    n_chunks: Optional[int] = None,
    n_chunk_max_cols: Optional[int] = None,
    serializer: Optional[MatrixSerializer] = None,
    randomizer_rng: Optional[np.random.Generator] = None,
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
    if randomizer_rng is not None:
        sources = list(sources)
        randomizer_rng.shuffle(sources)

    if n_chunks is None and n_chunk_max_cols is None:
        raise ValueError("exactly one of n_chunks and n_chunk_max_cols must be specified")

    if n_chunks is None:
        n_chunks = n_sources // n_chunk_max_cols
    elif n_chunk_max_cols is None:
        n_chunk_max_cols = int(np.ceil(n_sources / n_chunks))
    else:
        raise ValueError("exactly one of n_chunks and n_chunk_max_cols must be specified")

    os.makedirs(output_dir, exist_ok=True)

    if serializer is None:
        serializer = NumpySerializer("")

    snapshot_shape, snapshot_dtype = serializer.peek(sources[0])

    chunk_fnames = []
    i_source = 0
    for i_chunk in range(n_chunks):
        chunk_size = n_sources // n_chunks
        if i_chunk < n_sources % n_chunks:
            chunk_size += 1

        chunk = np.empty((np.prod(snapshot_shape), chunk_size), dtype=snapshot_dtype)
        for j, source in enumerate(sources[i_source:i_source + chunk_size]):
            chunk[:, j] = serializer.load(source).flatten()

        chunk_fname = os.path.join(output_dir, f"chunk_{i_chunk:04d}")
        chunk_fname = serializer.store(chunk, chunk_fname)
        chunk_fnames.append(chunk_fname)

        del chunk

        i_source += chunk_size

    return chunk_fnames


def get_cumulative_energy_ratios(s: np.ndarray) -> np.ndarray:
    """
    Compute cumulative, relative energy of the given array
    of singular values

    :param s: array of singular values, sorted desc
    :type s: np.ndarray
    :return: an array of same size as s
    :rtype: np.ndarray
    """
    if not len(s):
        return np.array([])

    return np.cumsum(s**2) / np.sum(s**2)


def get_truncation_rank(s: np.ndarray,
                        rank_max: Optional[int] = None,
                        magnitude_ratio_max: Optional[float] = None,
                        res_energy_ratio_max: Optional[float] = None) -> int:
    """
    Compute the appropriate truncation rank, taken as the tightest
    of the specified thresholds

    :param s: array of singular values, sorted desc
    :type s: np.ndarray
    :param rank_max: maximum number of singular values, defaults to None
    :type rank_max: Optional[int], optional
    :param magnitude_ratio_max: discard singular values whose relative magnitude is lower than the given value, defaults to None
    :type magnitude_ratio_max: Optional[float], optional
    :param res_energy_ratio_max: discard singular values whose cumulative relative energy is lower than the given value, defaults to None
    :type res_energy_ratio_max: Optional[float], optional
    :return: the minimum truncation rank
    :rtype: int
    """
    rmax = len(s)
    if not rmax:
        raise ValueError("empty singular values array")

    if rank_max is not None:
        rmax = min(rmax, rank_max)
    if magnitude_ratio_max is not None:
        r = np.searchsorted(np.flip(s) / np.max(s), magnitude_ratio_max)
        rmax = min(rmax, len(s) - r)
    if res_energy_ratio_max is not None:
        e = get_cumulative_energy_ratios(s)
        r = np.searchsorted(np.flip(1 - e), res_energy_ratio_max)
        rmax = min(rmax, len(s) - r)

    return max(1, rmax)


def get_pod(X: np.ndarray, rank_max: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return the truncated modes matrix and singular values of X using numpy.linalg.svd

    :param X: a 2d array to be decomposed, assumed X.shape[0] >= X.shape[1]
    :type X: np.ndarray
    :param rank_max: maximum number of singular values, defaults to None
    :type rank_max: Optional[int], optional
    :return: U 2d matrix of modes, s array of singular values
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    U, s, _ = np.linalg.svd(X, full_matrices=False)
    if rank_max is None:
        rank_max = len(s)

    return U[:, :rank_max], s[:rank_max]


def randomized_POD(
    sources: List[Union[str, np.ndarray]],
    rank_max: int,
    serializer: Optional[MatrixSerializer] = None,
    randomizer_rng: Optional[np.random.Generator] = None,
):
    shape, dtype = serializer.peek(sources[0])
    n_rows = math.prod(shape)
    n_cols = len(sources)
    random_samples = randomizer_rng.choice(n_cols, rank_max, replace=False)

    Z = np.empty((n_rows, rank_max), dtype=dtype)
    for i, j in enumerate(random_samples):
        Z[:, i] = serializer.load(sources[j]).flatten()

    Q, _ = np.linalg.qr(Z)
    Y = np.empty((rank_max, n_cols), dtype=dtype)
    for i, source in enumerate(sources):
        Y[:, i] = Q.T @ serializer.load(source).flatten()

    Uy, s, _ = np.linalg.svd(Y, full_matrices=False)
    U = Q @ Uy

    return U, s
