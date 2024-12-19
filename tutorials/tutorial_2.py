import os
import sys
import time
import shutil
import tempfile

os.environ["OMP_NUM_THREADS"] = "8"
import numpy as np

import hapod as hp

work_dir = "/scratch/lfabris/hapod_test"
hapod_tmp_dir = os.path.join(work_dir, "tmp")

snapshots_dir = os.path.join(work_dir, "snapshots")
os.makedirs(snapshots_dir, exist_ok=True)

n_rows = 3600000
n_max_svd_cols = 130
n_cols = n_max_svd_cols * 2
n_chunk_max_cols = n_max_svd_cols // 2

print(f"simulating snapshot matrix with size {(n_rows, n_cols)}")
print(
    f"storing {hp.get_matrix_memory_footprint((n_rows, n_cols)) / 2**30:.3f} GB worth of column snapshots"
)
print(f"chunks will collect {n_chunk_max_cols} columns each")

rng = np.random.default_rng()

elapsed_snapshots = -time.perf_counter()
snapshots_fnames = []
for i in range(n_cols):
    snapshot_fname = os.path.join(snapshots_dir, f"snapshot_{i:04d}.npy")
    np.save(snapshot_fname, rng.random((n_rows, 1)))

    snapshots_fnames.append(snapshot_fname)
elapsed_snapshots += time.perf_counter()
print(f"created {len(snapshots_fnames)} snapshot files in {elapsed_snapshots:.3f}")

chunks_dir = os.path.join(work_dir, "chunks")

elapsed_chunks = -time.perf_counter()
chunks_fnames = hp.make_chunks(
    snapshots_fnames,
    chunks_dir,
    n_chunk_max_cols=n_chunk_max_cols,
)
elapsed_chunks += time.perf_counter()
print(f"created {len(chunks_fnames)} column chunks files in {elapsed_chunks:.3f}")

elapsed_hapod = -time.perf_counter()
Uu, ss = hp.hapod(chunks_fnames,
                  rank_max=n_chunk_max_cols,
                  temp_work_dir=hapod_tmp_dir,
                  skip_last_truncation=True,
                  verbose=True)
elapsed_hapod += time.perf_counter()

print(f"finished hapod in {elapsed_hapod:.3f}")
print(f"    U.shape {Uu.shape}")
print(f"    ss.shape {ss.shape}")

np.save(os.path.join(work_dir, "U.npy"), Uu)
np.save(os.path.join(work_dir, "s.npy"), ss)

#shutil.rmtree(snapshots_dir)
#shutil.rmtree(chunks_dir)
