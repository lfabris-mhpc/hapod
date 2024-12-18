# %%
import os
import time

os.environ["OMP_NUM_THREADS"] = "8"
import numpy as np

import hapod as hp

# %%
n_rows = 3600000
n_cols = 136

# %%

ram_avail = hp.ram_size() / 2**30
print(f"available ram {ram_avail:.2f}")

ram_matrix = hp.matrix_memory_footprint((n_rows, n_cols)) / 2**30
print(f"matrix ram {ram_matrix:.2f}")

ram_svd = hp.svd_memory_footprint((n_rows, n_cols)) / 2**30
print(f"svd ram {ram_svd:.2f}")

# %%
assert ram_svd < ram_avail

np.random.seed(42)

elapsed_matrix = -time.perf_counter()
X = hp.random_matrix(n_rows, n_cols, n_cols)
elapsed_matrix += time.perf_counter()
print(f"matrix creation took {elapsed_matrix:.3f}")

elapsed_svd = -time.perf_counter()
U, s, VT = np.linalg.svd(X, full_matrices=False)
elapsed_svd += time.perf_counter()
print(f"svd computation took {elapsed_svd:.3f}")
