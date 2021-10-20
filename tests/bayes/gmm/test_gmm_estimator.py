# pylint: disable=missing-function-docstring
import math
import pytest
import pytorch_lightning as pl
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


@pytest.mark.parametrize("batch_size", [2, 4])
def test_fit_num_iter(batch_size: int):
    pl.seed_everything(0)

    # For the following data, K-means will find centroids [0.5, 3.5]. The estimator needs to find
    # the covariance (first iteration), then a new NLL is obtained (second iteration), and
    # finally, there is no improvmement in the NLL (third iteration).
    data = torch.as_tensor([[0.0], [1.0], [3.0], [4.0]])
    estimator = GaussianMixture(
        2,
        convergence_tolerance=0,
        batch_size=batch_size,
        trainer_params=dict(precision=64),
    )
    estimator.fit(data)

    assert estimator.num_iter_ == 3


@pytest.mark.parametrize(
    ("batch_size", "max_epochs", "converged"),
    [(2, 1, False), (2, 3, True), (4, 1, False), (4, 3, True)],
)
def test_fit_converged(batch_size: int, max_epochs: int, converged: bool):
    # pl.seed_everything(0)
    data = torch.as_tensor([[0.0], [1.0], [3.0], [4.0]])

    estimator = GaussianMixture(
        2,
        convergence_tolerance=0,
        batch_size=batch_size,
        trainer_params=dict(precision=64, max_epochs=max_epochs),
    )
    estimator.fit(data)
    assert estimator.converged_ == converged


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
    # pl.seed_everything(0)

    data = sample_gmm(
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
    print(estimator.model_.means)
    print(estimator.model_.precisions_cholesky)

    # Sklearn
    gmm = SklearnGaussianMixture(num_components, covariance_type=covariance_type)
    sklearn_nll = gmm.fit(data.numpy()).score(data.numpy())
    print(gmm.means_)
    print(gmm.precisions_cholesky_)

    print(ours_nll, sklearn_nll)

    assert math.isclose(ours_nll, -sklearn_nll, rel_tol=0.01, abs_tol=0.01)
