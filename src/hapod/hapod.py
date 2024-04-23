from typing import Optional, List, Tuple, Union
import time

import numpy as np

def get_cumulative_energy_frac(s: np.ndarray) -> np.ndarray:
    return np.cumsum(s**2) / np.sum(s**2)

def get_truncation_rank(s: np.ndarray,
                    rank_max: Optional[int]=None,
                    magnitude_frac_max: Optional[float]=None,
                    res_energy_frac_max: Optional[float]=None) -> float:
    rmax = len(s)

    if rank_max is not None:
        rmax = min(rmax, rank_max)
    if magnitude_frac_max is not None:
        r = np.searchsorted(np.flip(s)/np.max(s), magnitude_frac_max)
        rmax = min(rmax, len(s) - r)
    if res_energy_frac_max is not None:
        e = get_cumulative_energy_frac(s)
        r = np.searchsorted(np.flip(1-e), res_energy_frac_max)
        rmax = min(rmax, len(s) - r)

    return max(1, rmax)

def get_truncated_svd(A: np.ndarray,
                      rank_max: Optional[int]=None,
                      magnitude_frac_max: Optional[float]=None,
                      res_energy_frac_max: Optional[float]=None) -> Tuple[np.ndarray, np.ndarray]:
    U, s, _ = np.linalg.svd(A, full_matrices=False)
    rmax = get_truncation_rank(s,
                            rank_max=rank_max,
                            magnitude_frac_max=magnitude_frac_max,
                            res_energy_frac_max=res_energy_frac_max)

    return U[:, :rmax], s[:rmax]

def get_column_batches(X: np.ndarray, batch: int, debug: bool=False) -> List[np.ndarray]:
    Xs = []
    n_batches = int(np.ceil(X.shape[-1] / batch))
    size = int(np.floor(X.shape[-1] / n_batches))
    rest = X.shape[-1] - n_batches * size
    if debug:
        print(f"len {X.shape[-1]} batch {batch}")
        print(f"n_batches {n_batches} rest {rest} size {size}")
    assert n_batches * size + rest == X.shape[-1]

    start = 0
    for i in range(int(np.ceil(X.shape[-1] / batch))):
        end = start + size
        if i < rest:
            end += 1
        Xs.append(X[:, start:end])
        start = end

    return Xs

def hapod(Xs: List[Union[np.ndarray, str]] ,
            rank_max: Optional[int]=None,
            magnitude_frac_max: Optional[float]=None,
            res_energy_frac_max: Optional[float]=None,
            svd_impl=get_truncated_svd,
            debug: bool=False) -> Tuple[np.ndarray, np.ndarray]:
    if res_energy_frac_max is not None:
        if debug:
            print(f"target res_energy_frac_max {res_energy_frac_max}")
        h = np.floor(1 + np.log2(len(Xs)))
        res_energy_frac_max = 1 - np.power(1 - res_energy_frac_max, 1/h)
        if debug:
            print(f"estimated height {h}")
            print(f"actual res_energy_frac_max {res_energy_frac_max}")

    while len(Xs) > 1:
        X1, X2 = Xs.pop(), Xs.pop()
        if isinstance(X1, str):
            X1 = np.load(X1, mmap_mode="r")
        if isinstance(X2, str):
            X2 = np.load(X2, mmap_mode="r")

        elapsed_svd = -time.perf_counter()
        Uu, ss = svd_impl(np.hstack((X1, X2)),
                        rank_max=rank_max,
                        magnitude_frac_max=magnitude_frac_max,
                        res_energy_frac_max=res_energy_frac_max)
        elapsed_svd += time.perf_counter()
        if debug:
            print(f"U_svd.shape {Uu.shape}")
            print(f"elapsed_svd {elapsed_svd:.3f} s")

        if not len(Xs):
            return Uu, ss
        Xs.insert(0, Uu * ss[np.newaxis, :])

        if debug:
            print("Xs")
            for x in Xs:
                if isinstance(x, str):
                    print(f"    {x}")
                else:
                    print(f"    {x.shape}")
