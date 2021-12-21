# pylint: disable=missing-function-docstring
import math
from typing import Optional
import pytest
import torch
from sklearn.mixture import GaussianMixture as SklearnGaussianMixture  # type: ignore
from tests._data.gmm import sample_gmm
from pycave.bayes import GaussianMixture
from pycave.bayes.core import CovarianceType


def test_fit_model_config():
    estimator = GaussianMixture()
    data = torch.randn(1000, 4)
    estimator.fit(data)

    assert estimator.model_.config.num_components == 1
    assert estimator.model_.config.num_features == 4


@pytest.mark.parametrize("batch_size", [2, None])
def test_fit_num_iter(batch_size: Optional[int]):
    # For the following data, K-means will find centroids [0.5, 3.5]. The estimator first computes
    # the NLL (first iteration), afterwards there is no improvmement in the NLL (second iteration).
    data = torch.as_tensor([[0.0], [1.0], [3.0], [4.0]])
    estimator = GaussianMixture(
        2,
        batch_size=batch_size,
        trainer_params=dict(precision=64),
    )
    estimator.fit(data)

    assert estimator.num_iter_ == 2


@pytest.mark.flaky(max_runs=3, min_passes=1)
@pytest.mark.parametrize(
    ("batch_size", "max_epochs", "converged"),
    [(2, 1, False), (2, 3, True), (None, 1, False), (None, 3, True)],
)
def test_fit_converged(batch_size: Optional[int], max_epochs: int, converged: bool):
    data = torch.as_tensor([[0.0], [1.0], [3.0], [4.0]])

    estimator = GaussianMixture(
        2,
        batch_size=batch_size,
        trainer_params=dict(precision=64, max_epochs=max_epochs),
    )
    estimator.fit(data)
    assert estimator.converged_ == converged


@pytest.mark.flaky(max_runs=10, min_passes=1)
@pytest.mark.parametrize(
    ("num_datapoints", "batch_size", "num_features", "num_components", "covariance_type"),
    [
        (10000, 10000, 4, 4, "spherical"),
        (10000, 10000, 4, 4, "diag"),
        (10000, 10000, 4, 4, "tied"),
        (10000, 10000, 4, 4, "full"),
        (10000, 1000, 4, 4, "spherical"),
        (10000, 1000, 4, 4, "diag"),
        (10000, 1000, 4, 4, "tied"),
        (10000, 1000, 4, 4, "full"),
    ],
)
def test_fit_nll(
    num_datapoints: int,
    batch_size: int,
    num_features: int,
    num_components: int,
    covariance_type: CovarianceType,
):
    data, _ = sample_gmm(
        num_datapoints=num_datapoints,
        num_features=num_features,
        num_components=num_components,
        covariance_type=covariance_type,
    )

    # Ours
    estimator = GaussianMixture(
        num_components,
        covariance_type=covariance_type,
        batch_size=batch_size,
        trainer_params=dict(precision=64),
    )
    ours_nll = estimator.fit(data).score(data)

    # Sklearn
    gmm = SklearnGaussianMixture(num_components, covariance_type=covariance_type)
    sklearn_nll = gmm.fit(data.numpy()).score(data.numpy())

    # assert math.isclose(ours_nll, -sklearn_nll, rel_tol=0.01, abs_tol=0.01)
