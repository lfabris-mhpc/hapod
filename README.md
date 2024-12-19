# hapod
Simple implementation of Hierarchical Approximate Proper Orthogonal Decomposition in Python

This package is designed to approximate the left singular vectors and singular values of the snapshots matrix of a large dataset.

The assumptions are
- a single machine is available
- the snapshots are stored on disk as multiple files, and cannot be loaded at once in memory
- the Singular Value Decomposition primitive needs to load a dense matrix in memory

The implements the approach from [this paper](https://doi.org/10.1137/16M1085413) ([preprint](https://arxiv.org/abs/1607.05210)) and requires only numpy.

This package implements HAPOD by storing the intermediate results on disk.
The snapshots are initially grouped to store chunks of the full matrix on disk.
Iteratively, a pair of chunks is concatenated and SVD is performed; the results are truncated and used to reconstruct a new chunk, which is stored again.
The chunk size is chosen so that, during a merge, the SVD memory usage remains under a limit specified by the user.
The end results approximate the POD of the full snapshots matrix, truncated to the chunk size.
