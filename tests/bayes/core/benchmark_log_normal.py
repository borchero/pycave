# pylint: disable=missing-function-docstring
import numpy as np
import torch
from pytest_benchmark.fixture import BenchmarkFixture  # type: ignore
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky  # type: ignore
from sklearn.mixture._gaussian_mixture import _estimate_log_gaussian_prob  # type: ignore
from torch.distributions import MultivariateNormal
from torchgmm.bayes.core import cholesky_precision, log_normal


def test_log_normal_spherical(benchmark: BenchmarkFixture):
    data = torch.randn(10000, 100)
    means = torch.randn(50, 100)
    precisions = cholesky_precision(torch.rand(50), "spherical")
    benchmark(log_normal, data, means, precisions, covariance_type="spherical")


def test_torch_log_normal_spherical(benchmark: BenchmarkFixture):
    data = torch.randn(10000, 100)
    means = torch.randn(50, 100)
    covars = torch.rand(50)
    covar_matrices = torch.stack([torch.eye(means.size(-1)) * c for c in covars])

    cholesky = torch.linalg.cholesky(covar_matrices)
    distribution = MultivariateNormal(means, scale_tril=cholesky, validate_args=False)
    benchmark(distribution.log_prob, data.unsqueeze(1))


def test_numpy_log_normal_spherical(benchmark: BenchmarkFixture):
    data = np.random.randn(10000, 100)
    means = np.random.randn(50, 100)
    covars = np.random.rand(50)
    benchmark(_estimate_log_gaussian_prob, data, means, covars, "spherical")  # type: ignore


# -------------------------------------------------------------------------------------------------


def test_log_normal_diag(benchmark: BenchmarkFixture):
    data = torch.randn(10000, 100)
    means = torch.randn(50, 100)
    precisions = cholesky_precision(torch.rand(50, 100), "diag")
    benchmark(log_normal, data, means, precisions, covariance_type="diag")


def test_torch_log_normal_diag(benchmark: BenchmarkFixture):
    data = torch.randn(10000, 100)
    means = torch.randn(50, 100)
    covars = torch.rand(50, 100)
    covar_matrices = torch.stack([torch.diag(c) for c in covars])

    cholesky = torch.linalg.cholesky(covar_matrices)
    distribution = MultivariateNormal(means, scale_tril=cholesky, validate_args=False)
    benchmark(distribution.log_prob, data.unsqueeze(1))


def test_numpy_log_normal_diag(benchmark: BenchmarkFixture):
    data = np.random.randn(10000, 100)
    means = np.random.randn(50, 100)
    covars = np.random.rand(50, 100)
    benchmark(_estimate_log_gaussian_prob, data, means, covars, "diag")  # type: ignore


# -------------------------------------------------------------------------------------------------


def test_log_normal_full(benchmark: BenchmarkFixture):
    data = torch.randn(10000, 100)
    means = torch.randn(50, 100)
    A = torch.randn(50, 1000, 100)
    covars = A.permute(0, 2, 1).bmm(A)
    precisions = cholesky_precision(covars, "full")
    benchmark(log_normal, data, means, precisions, covariance_type="full")


def test_torch_log_normal_full(benchmark: BenchmarkFixture):
    data = torch.randn(10000, 100)
    means = torch.randn(50, 100)
    A = torch.randn(50, 1000, 100)
    covars = A.permute(0, 2, 1).bmm(A)

    cholesky = torch.linalg.cholesky(covars)
    distribution = MultivariateNormal(means, scale_tril=cholesky, validate_args=False)
    benchmark(distribution.log_prob, data.unsqueeze(1))


def test_numpy_log_normal_full(benchmark: BenchmarkFixture):
    data = np.random.randn(10000, 100)
    means = np.random.randn(50, 100)
    A = np.random.randn(50, 1000, 100)
    covars = np.matmul(np.transpose(A, (0, 2, 1)), A)

    precisions = _compute_precision_cholesky(covars, "full")  # type: ignore
    benchmark(
        _estimate_log_gaussian_prob,  # type: ignore
        data,
        means,
        precisions,
        covariance_type="full",
    )


# -------------------------------------------------------------------------------------------------


def test_log_normal_tied(benchmark: BenchmarkFixture):
    data = torch.randn(10000, 100)
    means = torch.randn(50, 100)
    A = torch.randn(1000, 100)
    covars = A.t().mm(A)
    precisions = cholesky_precision(covars, "tied")
    benchmark(log_normal, data, means, precisions, covariance_type="tied")


def test_torch_log_normal_tied(benchmark: BenchmarkFixture):
    data = torch.randn(10000, 100)
    means = torch.randn(50, 100)
    A = torch.randn(1000, 100)
    covars = A.t().mm(A)

    cholesky = torch.linalg.cholesky(covars)
    distribution = MultivariateNormal(means, scale_tril=cholesky, validate_args=False)
    benchmark(distribution.log_prob, data.unsqueeze(1))


def test_numpy_log_normal_tied(benchmark: BenchmarkFixture):
    data = np.random.randn(10000, 100)
    means = np.random.randn(50, 100)
    A = np.random.randn(1000, 100)
    covars = A.T.dot(A)

    precisions = _compute_precision_cholesky(covars, "tied")  # type: ignore
    benchmark(
        _estimate_log_gaussian_prob,  # type: ignore
        data,
        means,
        precisions,
        covariance_type="tied",
    )
