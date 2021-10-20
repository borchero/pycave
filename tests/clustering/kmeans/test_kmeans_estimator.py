# pylint: disable=missing-function-docstring
import math
from typing import Optional
import pytest
import pytorch_lightning as pl
import torch
from sklearn.cluster import KMeans as SklearnKMeans  # type: ignore
from tests._data.gmm import sample_gmm
from pycave.clustering import KMeans


def test_recover():
    torch.manual_seed(42)

    estimator = KMeans(2)
    data = torch.cat([torch.randn(1000, 4) * 0.1 - 1, torch.randn(1000, 4) * 0.1 + 1])
    estimator.fit(data)

    ordering = estimator.model_.centroids[:, 0].argsort()
    expected = torch.as_tensor([[-1.0, -1.0, -1.0, -1.0], [1.0, 1.0, 1.0, 1.0]])[ordering]

    assert torch.allclose(estimator.model_.centroids, expected, atol=1e-2)


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
    pl.seed_everything(0)

    data = sample_gmm(
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
