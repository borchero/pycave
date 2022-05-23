# pylint: disable=missing-function-docstring
from typing import Optional
import pytest
import pytorch_lightning as pl
import torch
from pytest_benchmark.fixture import BenchmarkFixture  # type: ignore
from sklearn.mixture import GaussianMixture as SklearnGaussianMixture  # type: ignore
from pycave.bayes import GaussianMixture
from pycave.bayes.core.types import CovarianceType
from tests._data.gmm import sample_gmm


@pytest.mark.parametrize(
    ("num_datapoints", "num_features", "num_components", "covariance_type"),
    [
        (10000, 8, 4, "diag"),
        (10000, 8, 4, "tied"),
        (10000, 8, 4, "full"),
        (100000, 32, 16, "diag"),
        (100000, 32, 16, "tied"),
        (100000, 32, 16, "full"),
        (1000000, 64, 64, "diag"),
    ],
)
def test_sklearn(
    benchmark: BenchmarkFixture,
    num_datapoints: int,
    num_features: int,
    num_components: int,
    covariance_type: CovarianceType,
):
    pl.seed_everything(0)
    data, means = sample_gmm(num_datapoints, num_features, num_components, covariance_type)

    estimator = SklearnGaussianMixture(
        num_components,
        covariance_type=covariance_type,
        tol=0,
        n_init=1,
        max_iter=100,
        reg_covar=1e-3,
        init_params="random",
        means_init=means.numpy(),
    )
    benchmark(estimator.fit, data.numpy())


@pytest.mark.parametrize(
    ("num_datapoints", "num_features", "num_components", "covariance_type", "batch_size"),
    [
        (10000, 8, 4, "diag", None),
        (10000, 8, 4, "tied", None),
        (10000, 8, 4, "full", None),
        (100000, 32, 16, "diag", None),
        (100000, 32, 16, "tied", None),
        (100000, 32, 16, "full", None),
        (1000000, 64, 64, "diag", None),
        (10000, 8, 4, "diag", 1000),
        (10000, 8, 4, "tied", 1000),
        (10000, 8, 4, "full", 1000),
        (100000, 32, 16, "diag", 10000),
        (100000, 32, 16, "tied", 10000),
        (100000, 32, 16, "full", 10000),
        (1000000, 64, 64, "diag", 100000),
    ],
)
def test_pycave(
    benchmark: BenchmarkFixture,
    num_datapoints: int,
    num_features: int,
    num_components: int,
    covariance_type: CovarianceType,
    batch_size: Optional[int],
):
    pl.seed_everything(0)
    data, means = sample_gmm(num_datapoints, num_features, num_components, covariance_type)

    estimator = GaussianMixture(
        num_components,
        covariance_type=covariance_type,
        init_means=means,
        convergence_tolerance=0,
        covariance_regularization=1e-3,
        batch_size=batch_size,
        trainer_params=dict(max_epochs=100),
    )
    benchmark(estimator.fit, data)


@pytest.mark.parametrize(
    ("num_datapoints", "num_features", "num_components", "covariance_type", "batch_size"),
    [
        (10000, 8, 4, "diag", None),
        (10000, 8, 4, "tied", None),
        (10000, 8, 4, "full", None),
        (100000, 32, 16, "diag", None),
        (100000, 32, 16, "tied", None),
        (100000, 32, 16, "full", None),
        (1000000, 64, 64, "diag", None),
        (10000, 8, 4, "diag", 1000),
        (10000, 8, 4, "tied", 1000),
        (10000, 8, 4, "full", 1000),
        (100000, 32, 16, "diag", 10000),
        (100000, 32, 16, "tied", 10000),
        (100000, 32, 16, "full", 10000),
        (1000000, 64, 64, "diag", 100000),
        (1000000, 64, 64, "tied", 100000),
    ],
)
def test_pycave_gpu(
    benchmark: BenchmarkFixture,
    num_datapoints: int,
    num_features: int,
    num_components: int,
    covariance_type: CovarianceType,
    batch_size: Optional[int],
):
    # Initialize GPU
    torch.empty(1, device="cuda:0")

    pl.seed_everything(0)
    data, means = sample_gmm(num_datapoints, num_features, num_components, covariance_type)

    estimator = GaussianMixture(
        num_components,
        covariance_type=covariance_type,
        init_means=means,
        convergence_tolerance=0,
        covariance_regularization=1e-3,
        batch_size=batch_size,
        trainer_params=dict(max_epochs=100, gpus=1),
    )
    benchmark(estimator.fit, data)
