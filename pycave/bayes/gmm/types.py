from __future__ import annotations
from typing import Literal

GaussianMixtureInitStrategy = Literal["random", "kmeans"]
GaussianMixtureInitStrategy.__doc__ = """
Strategy for initializing the parameters of a Gaussian mixture model.

- **random**: Samples responsibilities of datapoints at random and subsequently initializes means
  and covariances from these.
- **kmeans**: Runs K-Means via :class:`pycave.clustering.KMeans` and uses the centroids as the
  initial component means. For computing the covariances, responsibilities are given as the
  one-hot cluster assignments.
"""
