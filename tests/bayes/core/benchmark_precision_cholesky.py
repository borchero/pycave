import numpy as np
import torch
from pytest_benchmark.fixture import BenchmarkFixture  # type: ignore
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky  # type: ignore
from pycave.bayes.core import cholesky_precision


def test_cholesky_precision_spherical(benchmark: BenchmarkFixture):
    covars = torch.rand(50)
    benchmark(cholesky_precision, covars, "spherical")


def test_numpy_cholesky_precision_spherical(benchmark: BenchmarkFixture):
    covars = np.random.rand(50)
    benchmark(_compute_precision_cholesky, covars, "spherical")  # type: ignore


# -------------------------------------------------------------------------------------------------


def test_cholesky_precision_tied(benchmark: BenchmarkFixture):
    A = torch.randn(10000, 100)
    covars = A.t().mm(A)
    benchmark(cholesky_precision, covars, "tied")


def test_numpy_cholesky_precision_tied(benchmark: BenchmarkFixture):
    A = np.random.randn(10000, 100)
    covars = np.dot(A.T, A)
    benchmark(_compute_precision_cholesky, covars, "tied")  # type: ignore


# -------------------------------------------------------------------------------------------------


def test_cholesky_precision_full(benchmark: BenchmarkFixture):
    A = torch.randn(50, 10000, 100)
    covars = A.permute(0, 2, 1).bmm(A)
    benchmark(cholesky_precision, covars, "full")


def test_numpy_cholesky_precision_full(benchmark: BenchmarkFixture):
    A = np.random.randn(50, 10000, 100)
    covars = np.matmul(np.transpose(A, (0, 2, 1)), A)
    benchmark(_compute_precision_cholesky, covars, "full")  # type: ignore
