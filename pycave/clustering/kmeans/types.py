from __future__ import annotations
from typing import Literal

KMeansInitStrategy = Literal["random", "kmeans++"]
KMeansInitStrategy.__doc__ = """
Strategy for initializing KMeans centroids.

- **random**: Centroids are sampled randomly from the data. This has complexity ``O(n)`` for ``n``
  datapoints.
- **kmeans++**: Centroids are computed iteratively. The first centroid is sampled randomly from
  the data. Subsequently, centroids are sampled from the remaining datapoints with probability
  proportional to ``D(x)^2`` where ``D(x)`` is the distance of datapoint ``x`` to the closest
  centroid chosen so far. This has complexity ``O(kn)`` for ``k`` clusters and ``n`` datapoints.
  If done on mini-batches, the complexity increases to ``O(k^2 n)``. 
"""
