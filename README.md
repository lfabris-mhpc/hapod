# hapod
Naive implementation of Hierarchical Approximate Proper Orthogonal Decomposition in Python

The module provides a basic implementation of the approach in [this paper](https://doi.org/10.1137/16M1085413) and requires only numpy.

The module enables to approximate the POD (intended as the pair of modes matrix and singular values array) using limited working memory.

This approach can be faster and more accurate than randomized SVD.
