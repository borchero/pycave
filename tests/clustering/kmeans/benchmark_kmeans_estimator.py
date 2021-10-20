# pylint: disable=missing-function-docstring
from typing import Optional
import pytest
import pytorch_lightning as pl
import torch
from pytest_benchmark.fixture import BenchmarkFixture  # type: ignore
from sklearn.cluster import KMeans as SklearnKMeans  # type: ignore
from tests._data.gmm import sample_gmm
from pycave.clustering import KMeans
from pycave.clustering.kmeans.types import KMeansInitStrategy


@pytest.mark.parametrize(
    ("num_datapoints", "num_features", "num_centroids", "num_iter", "init_strategy"),
    [
        (10000, 8, 4, 100, "k-means++"),
        (100000, 32, 16, 100, "k-means++"),
        (1000000, 64, 64, 100, "k-means++"),
        (10000, 8, 4, 100, "random"),
        (100000, 32, 16, 100, "random"),
        (1000000, 64, 64, 100, "random"),
    ],
)
def test_sklearn(
    benchmark: BenchmarkFixture,
    num_datapoints: int,
    num_features: int,
    num_centroids: int,
    num_iter: int,
    init_strategy: str,
):
    pl.seed_everything(0)
    data = sample_gmm(num_datapoints, num_features, num_centroids, "spherical")

    estimator = SklearnKMeans(
        num_centroids,
        algorithm="full",
        n_init=1,
        max_iter=num_iter,
        tol=0,
        init=init_strategy,
    )
    benchmark(estimator.fit, data.numpy())


@pytest.mark.parametrize(
    ("num_datapoints", "batch_size", "num_features", "num_centroids", "num_iter", "init_strategy"),
    [
        (10000, None, 8, 4, 100, "kmeans++"),
        (10000, 1000, 8, 4, 100, "kmeans++"),
        (100000, None, 32, 16, 100, "kmeans++"),
        (100000, 10000, 32, 16, 100, "kmeans++"),
        (1000000, None, 64, 64, 100, "kmeans++"),
        (1000000, 100000, 64, 64, 100, "kmeans++"),
        (10000, None, 8, 4, 100, "random"),
        (10000, 1000, 8, 4, 100, "random"),
        (100000, None, 32, 16, 100, "random"),
        (100000, 10000, 32, 16, 100, "random"),
        (1000000, None, 64, 64, 100, "random"),
        (1000000, 100000, 64, 64, 100, "random"),
    ],
)
def test_pycave(
    benchmark: BenchmarkFixture,
    num_datapoints: int,
    batch_size: Optional[int],
    num_features: int,
    num_centroids: int,
    num_iter: int,
    init_strategy: KMeansInitStrategy,
):
    pl.seed_everything(0)
    data = sample_gmm(num_datapoints, num_features, num_centroids, "spherical")

    estimator = KMeans(
        num_centroids,
        init_strategy=init_strategy,
        batch_size=batch_size,
        trainer_params=dict(max_epochs=num_iter),
    )
    benchmark(estimator.fit, data)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize(
    ("num_datapoints", "batch_size", "num_features", "num_centroids", "num_iter", "init_strategy"),
    [
        (10000, None, 8, 4, 100, "kmeans++"),
        (10000, 1000, 8, 4, 100, "kmeans++"),
        (100000, None, 32, 16, 100, "kmeans++"),
        (100000, 10000, 32, 16, 100, "kmeans++"),
        (1000000, None, 64, 64, 100, "kmeans++"),
        (1000000, 100000, 64, 64, 100, "kmeans++"),
        (10000000, 1000000, 128, 128, 100, "kmeans++"),
        (100000000, 5000000, 512, 1024, 100, "kmeans++"),
        (10000, None, 8, 4, 100, "random"),
        (10000, 1000, 8, 4, 100, "random"),
        (100000, None, 32, 16, 100, "random"),
        (100000, 10000, 32, 16, 100, "random"),
        (1000000, None, 64, 64, 100, "random"),
        (1000000, 100000, 64, 64, 100, "random"),
        (10000000, 1000000, 128, 128, 100, "random"),
        (100000000, 5000000, 512, 1024, 100, "random"),
    ],
)
def test_pycave_gpu(
    benchmark: BenchmarkFixture,
    num_datapoints: int,
    batch_size: Optional[int],
    num_features: int,
    num_centroids: int,
    num_iter: int,
    init_strategy: KMeansInitStrategy,
):
    torch.cuda.init()

    pl.seed_everything(0)
    data = sample_gmm(num_datapoints, num_features, num_centroids, "spherical")

    estimator = KMeans(
        num_centroids,
        init_strategy=init_strategy,
        batch_size=batch_size,
        trainer_params=dict(gpus=1, max_epochs=num_iter),
    )
    benchmark(estimator.fit, data)
