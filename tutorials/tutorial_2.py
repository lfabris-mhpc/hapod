import os
import shutil
import tempfile

os.environ["OMP_NUM_THREADS"] = "8"
import numpy as np

import hapod as hp

work_dir = "/scratch/lfabris/hapod_test"

snapshots_dir = os.path.join(work_dir, "snapshots")
os.makedirs(snapshots_dir, exist_ok=True)

n_rows = 3600000
n_cols = 1000

rng = np.random.default_rng()

# X = hp.random_matrix(n_rows, n_cols, n_cols)

# snapshots_fnames = []
# for i, x in enumerate(np.array_split(X, n_cols, axis=1)):
#     snapshot_fname = os.path.join(work_dir, f"snapshot_{i:04d}.npy")
#     np.save(snapshot_fname, x)

#     snapshots_fnames.append(snapshot_fname)

# del X

snapshots_fnames = []
for i in range(n_cols):
    snapshot_fname = os.path.join(work_dir, f"snapshot_{i:04d}.npy")
    np.save(snapshot_fname, rng.random((n_rows, 1)))

    snapshots_fnames.append(snapshot_fname)

print(f"created {len(snapshots_fnames)} snapshot files")

chunks_dir = os.path.join(work_dir, "chunks")

chunks_fnames = hp.make_chunks(
    snapshots_fnames,
    chunks_dir,
    n_chunk_max_cols=20,
)

print(f"created {len(chunks_fnames)} column chunks files")

Uu, ss = hp.hapod(chunks_fnames, rank_max=11, verbose=False)

print(f"finished hapod")
print(f"    U.shape {Uu.shape}")
print(f"    ss.shape {ss.shape}")

# shutil.rmtree(snapshots_dir)
# shutil.rmtree(chunks_dir)
