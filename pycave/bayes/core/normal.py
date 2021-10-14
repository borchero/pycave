import torch
from ._jit import jit_log_normal, jit_sample_normal
from .types import CovarianceType


def cholesky_precision(covariances: torch.Tensor, covariance_type: CovarianceType) -> torch.Tensor:
    """
    Computes the Cholesky decompositions of the precision matrices induced by the provided
    covariance matrices.

    Args:
        covariances: A tensor of shape ``[num_components, dim, dim]``, ``[dim, dim]``,
            ``[num_components, dim]``, ``[dim]`` or ``[num_components]`` depending on the
            ``covariance_type``. These are the covariance matrices of multivariate Normal
            distributions.
        covariance_type: The type of covariance for the covariance matrices given.

    Returns:
        A tensor of the same shape as ``covariances``, providing the lower-triangular Cholesky
        decompositions of the precision matrices.
    """
    if covariance_type in ("tied", "full"):
        # Compute Cholesky decomposition
        cholesky = torch.linalg.cholesky(covariances)
        # Invert
        num_features = covariances.size(-1)
        target = torch.eye(num_features, dtype=covariances.dtype, device=covariances.device)
        if covariance_type == "full":
            num_components = covariances.size(0)
            target = target.unsqueeze(0).expand(num_components, -1, -1)
        return torch.triangular_solve(target, cholesky, upper=False).solution.transpose(-2, -1)

    # "Simple" kind of covariance
    return covariances.sqrt().reciprocal()


def log_normal(
    x: torch.Tensor,
    means: torch.Tensor,
    precisions_cholesky: torch.Tensor,
    covariance_type: CovarianceType,
) -> torch.Tensor:
    """
    Computes the log-probability of the given data for multiple multivariate Normal distributions
    defined by their means and covariances.

    Args:
        x: A tensor of shape ``[num_datapoints, dim]``. This is the data to compute the
            log-probability for.
        means: A tensor of shape ``[num_components, dim]``. These are the means of the multivariate
            Normal distributions.
        precisions_cholesky: A tensor of shape ``[num_components, dim, dim]``, ``[dim, dim]``,
            ``[num_components, dim]``, ``[dim]`` or ``[num_components]`` depending on the
            ``covariance_type``. These are the upper-triangular Cholesky matrices for the inverse
            covariance matrices (i.e. precision matrices) of the multivariate Normal distributions.
        covariance_type: The type of covariance for the covariance matrices given.

    Returns:
        A tensor of shape ``[num_datapoints, num_components]`` with the log-probabilities for each
        datapoint and each multivariate Normal distribution.
    """
    return jit_log_normal(x, means, precisions_cholesky, covariance_type)


def sample_normal(
    num: int,
    mean: torch.Tensor,
    cholesky_precisions: torch.Tensor,
    covariance_type: CovarianceType,
) -> torch.Tensor:
    """
    Samples the given number of times from the multivariate Normal distribution described by the
    mean and Cholesky precision.

    Args:
        num: The number of times to sample.
        means: A tensor of shape ``[dim]`` with the mean of the distribution to sample from.
        choleksy_precisions: A tensor of shape ``[dim, dim]``, ``[dim]``, ``[dim]`` or ``[1]``
            depending on the ``covariance_type``. This is the corresponding Cholesky precision
            matrix for the mean.
        covariance_type: The type of covariance for the covariance matrix given.

    Returns:
        A tensor of shape ``[num_samples, dim]`` with the samples from the Normal distribution.
    """
    return jit_sample_normal(num, mean, cholesky_precisions, covariance_type)
