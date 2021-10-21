# pylint: disable=missing-function-docstring
import math
from typing import Optional
import pytest
import torch
from sklearn.cluster import KMeans as SklearnKMeans  # type: ignore
from tests._data.gmm import sample_gmm
from pycave.clustering import KMeans


def test_fit_automatic_config():
    estimator = KMeans(4)
    data = torch.cat([torch.randn(1000, 3) * 0.1 - 100, torch.randn(1000, 3) * 0.1 + 100])
    estimator.fit(data)
    assert estimator.model_.config.num_clusters == 4
    assert estimator.model_.config.num_features == 3


def test_fit_num_iter():
    # The k-means++ iterations should find the centroids. Afterwards, it should only take a single
    # epoch until the centroids do not change anymore.
    data = torch.cat([torch.randn(1000, 4) * 0.1 - 100, torch.randn(1000, 4) * 0.1 + 100])

    estimator = KMeans(2)
    estimator.fit(data)

    assert estimator.num_iter_ == 1


@pytest.mark.flaky(max_runs=2, min_passes=1)
@pytest.mark.parametrize(
    ("num_epochs", "converged"),
    [(100, True), (1, False)],
)
def test_fit_converged(num_epochs: int, converged: bool):
    data, _ = sample_gmm(
        num_datapoints=10000,
        num_features=8,
        num_components=4,
        covariance_type="spherical",
    )

    estimator = KMeans(4, trainer_params=dict(max_epochs=num_epochs))
    estimator.fit(data)

    assert estimator.converged_ == converged


@pytest.mark.flaky(max_runs=2, min_passes=1)
@pytest.mark.parametrize(
    ("num_datapoints", "batch_size", "num_features", "num_centroids"),
    [
        (10000, None, 8, 4),
        (10000, 1000, 8, 4),
    ],
)
def test_fit_inertia(
    num_datapoints: int,
    batch_size: Optional[int],
    num_features: int,
    num_centroids: int,
):
    data, _ = sample_gmm(
        num_datapoints=num_datapoints,
        num_features=num_features,
        num_components=num_centroids,
        covariance_type="spherical",
    )

    # Ours
    estimator = KMeans(
        num_centroids,
        batch_size=batch_size,
        trainer_params=dict(precision=64),
    )
    ours_inertia = float("inf")
    for _ in range(10):
        ours_inertia = min(ours_inertia, estimator.fit(data).score(data))

    # Sklearn
    gmm = SklearnKMeans(num_centroids, n_init=10)
    sklearn_inertia = gmm.fit(data.numpy()).score(data.numpy())

    assert math.isclose(ours_inertia, -sklearn_inertia / data.size(0), rel_tol=0.01, abs_tol=0.01)
