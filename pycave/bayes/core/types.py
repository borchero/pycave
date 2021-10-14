from typing import Literal
import torch

CovarianceType = Literal["full", "tied", "diag", "spherical"]
"""
The type of covariance to use for a set of multivariate Normal distributions.

- `full`: Each distribution has a full covariance matrix.
- `tied`: All distributions share the same full covariance matrix.
- `diag`: Each distribution has a diagonal covariance matrix.
- `spherical`: Each distribution has a diagonal covariance matrix which is a multiple of the
    identity matrix.
"""


def covariance_dim(covariance_type: CovarianceType) -> int:
    """
    Returns the number of dimension of the covariance matrix for a set of components.

    Args:
        covariance_type: The type of covariance to obtain the dimension for.

    Returns:
        The number of dimensions.
    """
    if covariance_type == "full":
        return 3
    if covariance_type in ("tied", "diag"):
        return 2
    return 1


def covariance_shape(
    num_components: int, num_features: int, covariance_type: CovarianceType
) -> torch.Size:
    """
    Returns the expected shape of the covariance matrix for the given number of components with
    the provided number of features based on the covariance type.

    Args:
        num_components: The number of Normal distributions to describe with the covariance.
        num_features: The dimensionality of the Normal distributions.
        covariance_type: The type of covariance to use.

    Returns:
        The expected size of the tensor representing the covariances.
    """
    if covariance_type == "full":
        return torch.Size([num_components, num_features, num_features])
    if covariance_type == "tied":
        return torch.Size([num_features, num_features])
    if covariance_type == "diag":
        return torch.Size([num_components, num_features])
    # covariance_type == "spherical"
    return torch.Size([num_components])
