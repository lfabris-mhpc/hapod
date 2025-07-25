import platform
import subprocess
import math
from typing import Optional, Tuple

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
        result = subprocess.run(
            ["wmic", "computersystem", "get", "TotalPhysicalMemory"],
            stdout=subprocess.PIPE,
        )
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
    return (
        2.18
        * (math.prod(shape) + (math.prod(shape) // v_size) * v_size + v_size)
        * np.dtype(dtype).itemsize
    )


def get_randomized_svd_memory_footprint(
    shape: Tuple[int], n_samples: int, dtype: np.dtype = np.float64
) -> int:
    # return get_matrix_memory_footprint((shape[0], n_samples), dtype) \
    #     + get_matrix_memory_footprint((n_samples, n_samples), dtype) \
    #     + get_svd_memory_footprint((n_samples, shape[-1]), dtype)

    return 6.5 * get_matrix_memory_footprint(
        (shape[0], n_samples), dtype
    ) + get_svd_memory_footprint((n_samples, shape[-1]), dtype)


def get_max_svd_columns(
    n_rows: int, memory_limit: Optional[int] = None, dtype: np.dtype = np.float64
) -> int:
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


def get_max_svd_square(
    memory_limit: Optional[int] = None, dtype: np.dtype = np.float64
) -> int:
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


def get_max_randomized_svd_samples(
    shape: Tuple[int], memory_limit: Optional[int] = None, dtype: np.dtype = np.float64
) -> int:
    if memory_limit is None:
        memory_limit = get_memory_size()
    itemsize = np.dtype(dtype).itemsize

    n_rows, n_cols = shape

    lb = 1
    ub = int(memory_limit) // itemsize
    while (ub - lb) > 0:
        n_samples = (ub + lb + 1) // 2
        ram_req = get_randomized_svd_memory_footprint(
            (n_rows, n_cols), n_samples, dtype
        )

        # print(f"bounds {lb, ub}")
        # print(n_samples)
        if ram_req > memory_limit:
            ub = n_samples - 1
        elif ram_req < memory_limit:
            lb = n_samples
        else:
            return n_samples

    return lb


def get_n_chunks_fulltree(n_cols: int, n_chunk_max_cols: int) -> int:
    depth = int(np.ceil(np.log2(n_cols / n_chunk_max_cols)))
    return 2 ** max(0, depth)
