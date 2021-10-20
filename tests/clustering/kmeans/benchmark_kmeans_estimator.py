# pylint: disable=missing-function-docstring
from typing import Optional
import pytest
import pytorch_lightning as pl
import torch
from pytest_benchmark.fixture import BenchmarkFixture  # type: ignore
from sklearn.cluster import KMeans as SklearnKMeans  # type: ignore
from tests._data.gmm import sample_gmm
from pycave.clustering import KMeans


@pytest.mark.parametrize(
    ("num_datapoints", "num_features", "num_centroids"),
    [
        (10000, 8, 4),
        (100000, 32, 16),
        (1000000, 64, 64),
        # (10000000, 128, 128),
        # (1000000000, 512, 1024),
    ],
)
def test_sklearn(
    benchmark: BenchmarkFixture, num_datapoints: int, num_features: int, num_centroids: int
):
    pl.seed_everything(0)
    data = sample_gmm(num_datapoints, num_features, num_centroids, "spherical")

    estimator = SklearnKMeans(num_centroids, n_init=1)
    benchmark(estimator.fit, data.numpy())


@pytest.mark.parametrize(
    ("num_datapoints", "batch_size", "num_features", "num_centroids"),
    [
        (10000, None, 8, 4),
        (100000, None, 32, 16),
        (1000000, None, 64, 64),
        # (10000000, 128, 128),
        # (1000000000, 512, 1024),
    ],
)
def test_pycave(
    benchmark: BenchmarkFixture,
    num_datapoints: int,
    batch_size: Optional[int],
    num_features: int,
    num_centroids: int,
):
    pl.seed_everything(0)
    data = sample_gmm(num_datapoints, num_features, num_centroids, "spherical")

    estimator = KMeans(num_centroids, batch_size=batch_size)
    benchmark(estimator.fit, data)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize(
    ("num_datapoints", "batch_size", "num_features", "num_centroids"),
    [
        (10000, None, 8, 4),
        (100000, None, 32, 16),
        (1000000, None, 64, 64),
        # (10000000, 128, 128),
        # (1000000000, 512, 1024),
    ],
)
def test_pycave_gpu(
    benchmark: BenchmarkFixture,
    num_datapoints: int,
    batch_size: Optional[int],
    num_features: int,
    num_centroids: int,
):
    torch.cuda.init()

    pl.seed_everything(0)
    data = sample_gmm(num_datapoints, num_features, num_centroids, "spherical")

    estimator = KMeans(num_centroids, batch_size=batch_size, trainer_params=dict(gpus=1))
    benchmark(estimator.fit, data)
