# pylint: disable=missing-function-docstring
import math
import torch


def jit_log_normal(
    x: torch.Tensor,
    means: torch.Tensor,
    precisions_cholesky: torch.Tensor,
    covariance_type: str,
) -> torch.Tensor:
    if covariance_type == "full":
        # Precision shape is `[num_components, dim, dim]`.
        log_prob = x.new_empty((x.size(0), means.size(0)))
        # We loop here to not blow up the size of intermediate matrices
        for k, (mu, prec_chol) in enumerate(zip(means, precisions_cholesky)):
            inner = x.matmul(prec_chol) - mu.matmul(prec_chol)
            log_prob[:, k] = inner.square().sum(1)
    elif covariance_type == "tied":
        # Precision shape is `[dim, dim]`.
        a = x.matmul(precisions_cholesky)  # [N, D]
        b = means.matmul(precisions_cholesky)  # [K, D]
        log_prob = (a.unsqueeze(1) - b).square().sum(-1)
    else:
        precisions = precisions_cholesky.square()
        if covariance_type == "diag":
            # Precision shape is `[num_components, dim]`.
            x_prob = torch.matmul(x * x, precisions.t())
            m_prob = torch.einsum("ij,ij,ij->i", means, means, precisions)
            xm_prob = torch.matmul(x, (means * precisions).t())
        else:  # covariance_type == "spherical"
            # Precision shape is `[num_components]`
            x_prob = torch.ger(torch.einsum("ij,ij->i", x, x), precisions)
            m_prob = torch.einsum("ij,ij->i", means, means) * precisions
            xm_prob = torch.matmul(x, means.t() * precisions)

        log_prob = x_prob - 2 * xm_prob + m_prob

    num_features = x.size(1)
    logdet = _cholesky_logdet(num_features, precisions_cholesky, covariance_type)
    constant = math.log(2 * math.pi) * num_features
    return logdet - 0.5 * (constant + log_prob)


def _cholesky_logdet(
    num_features: int,
    precisions_cholesky: torch.Tensor,
    covariance_type: str,
) -> torch.Tensor:
    if covariance_type == "full":
        return precisions_cholesky.diagonal(dim1=-2, dim2=-1).log().sum(-1)
    if covariance_type == "tied":
        return precisions_cholesky.diagonal().log().sum(-1)
    if covariance_type == "diag":
        return precisions_cholesky.log().sum(1)
    # covariance_type == "spherical"
    return precisions_cholesky.log() * num_features


# -------------------------------------------------------------------------------------------------


def jit_sample_normal(
    num: int,
    mean: torch.Tensor,
    cholesky_precisions: torch.Tensor,
    covariance_type: str,
) -> torch.Tensor:
    samples = torch.randn(num, mean.size(0), dtype=mean.dtype, device=mean.device)
    chol_covariance = _cholesky_covariance(cholesky_precisions, covariance_type)

    if covariance_type in ("tied", "full"):
        scale = chol_covariance.matmul(samples.unsqueeze(-1)).squeeze(-1)
    else:
        scale = chol_covariance * samples

    return mean + scale


def _cholesky_covariance(chol_precision: torch.Tensor, covariance_type: str) -> torch.Tensor:
    # For complex covariance types, invert the
    if covariance_type in ("tied", "full"):
        num_features = chol_precision.size(-1)
        target = torch.eye(num_features, dtype=chol_precision.dtype, device=chol_precision.device)
        return torch.triangular_solve(target, chol_precision, upper=True).solution.t()

    # Simple covariance type
    return chol_precision.reciprocal()
