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
    ("num_datapoints", "num_features", "num_centroids", "num_iter"),
    [
        (10000, 8, 4, 50),
        (100000, 32, 16, 100),
        (1000000, 64, 64, 150),
    ],
)
def test_sklearn(
    benchmark: BenchmarkFixture,
    num_datapoints: int,
    num_features: int,
    num_centroids: int,
    num_iter: int,
):
    pl.seed_everything(0)
    data = sample_gmm(num_datapoints, num_features, num_centroids, "spherical")

    estimator = SklearnKMeans(num_centroids, algorithm="full", n_init=1, max_iter=num_iter, tol=0)
    benchmark(estimator.fit, data.numpy())


@pytest.mark.parametrize(
    ("num_datapoints", "batch_size", "num_features", "num_centroids", "num_iter"),
    [
        (10000, None, 8, 4, 50),
        (10000, 1000, 8, 4, 50),
        (100000, None, 32, 16, 100),
        (100000, 10000, 32, 16, 100),
        (1000000, None, 64, 64, 150),
        (1000000, 100000, 64, 64, 150),
    ],
)
def test_pycave(
    benchmark: BenchmarkFixture,
    num_datapoints: int,
    batch_size: Optional[int],
    num_features: int,
    num_centroids: int,
    num_iter: int,
):
    pl.seed_everything(0)
    data = sample_gmm(num_datapoints, num_features, num_centroids, "spherical")

    estimator = KMeans(
        num_centroids,
        batch_size=batch_size,
        trainer_params=dict(max_epochs=num_iter),
    )
    benchmark(estimator.fit, data)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize(
    ("num_datapoints", "batch_size", "num_features", "num_centroids", "num_iter"),
    [
        (10000, None, 8, 4, 50),
        (10000, 1000, 8, 4, 50),
        (100000, None, 32, 16, 100),
        (100000, 10000, 32, 16, 100),
        (1000000, None, 64, 64, 150),
        (1000000, 100000, 64, 64, 150),
        (10000000, None, 128, 128, 200),
        (10000000, 1000000, 128, 128, 200),
        (1000000000, 10000000, 512, 1024, 250),
    ],
)
def test_pycave_gpu(
    benchmark: BenchmarkFixture,
    num_datapoints: int,
    batch_size: Optional[int],
    num_features: int,
    num_centroids: int,
    num_iter: int,
):
    torch.cuda.init()

    pl.seed_everything(0)
    data = sample_gmm(num_datapoints, num_features, num_centroids, "spherical")

    estimator = KMeans(
        num_centroids,
        batch_size=batch_size,
        trainer_params=dict(gpus=1, max_epochs=num_iter),
    )
    benchmark(estimator.fit, data)
