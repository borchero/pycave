from __future__ import annotations
from typing import Literal

CovarianceType = Literal["full", "tied", "diag", "spherical"]
CovarianceType.__doc__ = """
The type of covariance to use for a set of multivariate Normal distributions.

- **full**: Each distribution has a full covariance matrix. Covariance matrix is a tensor of shape
  ``[num_components, num_features, num_features]``.
- **tied**: All distributions share the same full covariance matrix. Covariance matrix is a tensor
  of shape ``[num_features, num_features]``.
- **diag**: Each distribution has a diagonal covariance matrix. Covariance matrix is a tensor of
  shape ``[num_components, num_features]``.
- **spherical**: Each distribution has a diagonal covariance matrix which is a multiple of the
  identity matrix. Covariance matrix is a tensor of shape ``[num_components]``.
"""
