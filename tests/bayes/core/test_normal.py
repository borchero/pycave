# pylint: disable=missing-function-docstring
import pytest
import torch
from sklearn.mixture._gaussian_mixture import _compute_log_det_cholesky  # type: ignore
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky  # type: ignore
from torch.distributions import MultivariateNormal
from pycave.bayes.core import cholesky_precision, covariance, log_normal, sample_normal
from pycave.bayes.core._jit import _cholesky_logdet  # type: ignore
from tests._data.normal import (
    sample_data,
    sample_diag_covars,
    sample_full_covars,
    sample_means,
    sample_spherical_covars,
)

# -------------------------------------------------------------------------------------------------
# CHOLESKY PRECISIONS
# -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("covars", sample_spherical_covars([70, 5, 200]))
def test_cholesky_precision_spherical(covars: torch.Tensor):
    expected = _compute_precision_cholesky(covars.numpy(), "spherical")  # type: ignore
    actual = cholesky_precision(covars, "spherical")
    assert torch.allclose(torch.as_tensor(expected, dtype=torch.float), actual)


@pytest.mark.parametrize("covars", sample_diag_covars([70, 5, 200], [3, 50, 100]))
def test_cholesky_precision_diag(covars: torch.Tensor):
    expected = _compute_precision_cholesky(covars.numpy(), "diag")  # type: ignore
    actual = cholesky_precision(covars, "diag")
    assert torch.allclose(torch.as_tensor(expected, dtype=torch.float), actual)


@pytest.mark.parametrize("covars", sample_full_covars([70, 5, 200], [3, 50, 100]))
def test_cholesky_precision_full(covars: torch.Tensor):
    expected = _compute_precision_cholesky(covars.numpy(), "full")  # type: ignore
    actual = cholesky_precision(covars, "full")
    assert torch.allclose(torch.as_tensor(expected, dtype=torch.float), actual)


@pytest.mark.parametrize("covars", sample_full_covars([1, 1, 1], [3, 50, 100]))
def test_cholesky_precision_tied(covars: torch.Tensor):
    expected = _compute_precision_cholesky(covars.numpy(), "tied")  # type: ignore
    actual = cholesky_precision(covars, "tied")
    assert torch.allclose(torch.as_tensor(expected, dtype=torch.float), actual)


# -------------------------------------------------------------------------------------------------
# COVARIANCES
# -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("covars", sample_spherical_covars([70, 5, 200]))
def test_covariances_spherical(covars: torch.Tensor):
    precision_cholesky = _compute_precision_cholesky(covars.numpy(), "spherical")  # type: ignore
    actual = covariance(torch.as_tensor(precision_cholesky, dtype=torch.float), "spherical")
    assert torch.allclose(covars, actual)


@pytest.mark.parametrize("covars", sample_diag_covars([70, 5, 200], [3, 50, 100]))
def test_covariances_diag(covars: torch.Tensor):
    precision_cholesky = _compute_precision_cholesky(covars.numpy(), "diag")  # type: ignore
    actual = covariance(torch.as_tensor(precision_cholesky, dtype=torch.float), "diag")
    assert torch.allclose(covars, actual)


@pytest.mark.parametrize("covars", sample_full_covars([70, 5, 200], [3, 50, 100]))
def test_covariances_full(covars: torch.Tensor):
    precision_cholesky = _compute_precision_cholesky(covars.numpy(), "full")  # type: ignore
    actual = covariance(torch.as_tensor(precision_cholesky, dtype=torch.double), "full")
    assert torch.allclose(covars.to(torch.double), actual, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("covars", sample_full_covars([1, 1, 1], [3, 50, 100]))
def test_covariances_tied(covars: torch.Tensor):
    precision_cholesky = _compute_precision_cholesky(covars.numpy(), "tied")  # type: ignore
    actual = covariance(torch.as_tensor(precision_cholesky, dtype=torch.double), "tied")
    assert torch.allclose(covars.to(torch.double), actual, rtol=1e-3, atol=1e-3)


# -------------------------------------------------------------------------------------------------
# CHOLESKY LOG DETERMINANTS
# -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("covars", sample_spherical_covars([70, 5, 200]))
def test_cholesky_logdet_spherical(covars: torch.Tensor):
    expected = _compute_log_det_cholesky(  # type: ignore
        _compute_precision_cholesky(covars.numpy(), "spherical"), "spherical", 100  # type: ignore
    )
    actual = _cholesky_logdet(  # type: ignore
        100,
        cholesky_precision(covars, "spherical"),
        "spherical",
    )
    assert torch.allclose(torch.as_tensor(expected, dtype=torch.float), actual)


@pytest.mark.parametrize("covars", sample_diag_covars([70, 5, 200], [3, 50, 100]))
def test_cholesky_logdet_diag(covars: torch.Tensor):
    expected = _compute_log_det_cholesky(  # type: ignore
        _compute_precision_cholesky(covars.numpy(), "diag"),  # type: ignore
        "diag",
        covars.size(1),
    )
    actual = _cholesky_logdet(  # type: ignore
        covars.size(1),
        cholesky_precision(covars, "diag"),
        "diag",
    )
    assert torch.allclose(torch.as_tensor(expected, dtype=torch.float), actual)


@pytest.mark.parametrize("covars", sample_full_covars([70, 5, 200], [3, 50, 100]))
def test_cholesky_logdet_full(covars: torch.Tensor):
    expected = _compute_log_det_cholesky(  # type: ignore
        _compute_precision_cholesky(covars.numpy(), "full"),  # type: ignore
        "full",
        covars.size(1),
    )
    actual = _cholesky_logdet(  # type: ignore
        covars.size(1),
        cholesky_precision(covars, "full"),
        "full",
    )
    assert torch.allclose(torch.as_tensor(expected, dtype=torch.float), actual)


@pytest.mark.parametrize("covars", sample_full_covars([1, 1, 1], [3, 50, 100]))
def test_cholesky_logdet_tied(covars: torch.Tensor):
    expected = _compute_log_det_cholesky(  # type: ignore
        _compute_precision_cholesky(covars.numpy(), "tied"),  # type: ignore
        "tied",
        covars.size(0),
    )
    actual = _cholesky_logdet(  # type: ignore
        covars.size(0),
        cholesky_precision(covars, "tied"),
        "tied",
    )
    assert torch.allclose(torch.as_tensor(expected, dtype=torch.float), actual)


# -------------------------------------------------------------------------------------------------
# LOG NORMAL
# -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "x, means, covars",
    zip(
        sample_data([10, 50, 100], [3, 50, 100]),
        sample_means([70, 5, 200], [3, 50, 100]),
        sample_spherical_covars([70, 5, 200]),
    ),
)
def test_log_normal_spherical(x: torch.Tensor, means: torch.Tensor, covars: torch.Tensor):
    covar_matrices = torch.stack([torch.eye(means.size(-1)) * c for c in covars])
    precisions_cholesky = cholesky_precision(covars, "spherical")
    actual = log_normal(x, means, precisions_cholesky, covariance_type="spherical")
    _assert_log_prob(actual, x, means, covar_matrices)


@pytest.mark.parametrize(
    "x, means, covars",
    zip(
        sample_data([10, 50, 100], [3, 50, 100]),
        sample_means([70, 5, 200], [3, 50, 100]),
        sample_diag_covars([70, 5, 200], [3, 50, 100]),
    ),
)
def test_log_normal_diag(x: torch.Tensor, means: torch.Tensor, covars: torch.Tensor):
    covar_matrices = torch.stack([torch.diag(c) for c in covars])
    precisions_cholesky = cholesky_precision(covars, "diag")
    actual = log_normal(x, means, precisions_cholesky, covariance_type="diag")
    _assert_log_prob(actual, x, means, covar_matrices)


@pytest.mark.parametrize(
    "x, means, covars",
    zip(
        sample_data([10, 50, 100], [3, 50, 100]),
        sample_means([70, 5, 200], [3, 50, 100]),
        sample_full_covars([70, 5, 200], [3, 50, 100]),
    ),
)
def test_log_normal_full(x: torch.Tensor, means: torch.Tensor, covars: torch.Tensor):
    precisions_cholesky = cholesky_precision(covars, "full")
    actual = log_normal(x, means, precisions_cholesky, covariance_type="full")
    _assert_log_prob(actual.float(), x, means, covars)


@pytest.mark.parametrize(
    "x, means, covars",
    zip(
        sample_data([10, 50, 100], [3, 50, 100]),
        sample_means([70, 5, 200], [3, 50, 100]),
        sample_full_covars([1, 1, 1], [3, 50, 100]),
    ),
)
def test_log_normal_tied(x: torch.Tensor, means: torch.Tensor, covars: torch.Tensor):
    precisions_cholesky = cholesky_precision(covars, "tied")
    actual = log_normal(x, means, precisions_cholesky, covariance_type="tied")
    _assert_log_prob(actual, x, means, covars)


# -------------------------------------------------------------------------------------------------
# SAMPLING
# -------------------------------------------------------------------------------------------------


@pytest.mark.flaky(max_runs=3, min_passes=1)
def test_sample_normal_spherical():
    mean = torch.tensor([1.5, 3.5])
    covar = torch.tensor(4.0)
    target_covar = torch.tensor([[4.0, 0.0], [0.0, 4.0]])

    n = 1_000_000
    precisions = cholesky_precision(covar, "spherical")
    samples = sample_normal(n, mean, precisions, "spherical")

    sample_mean = samples.mean(0)
    sample_covar = (samples - sample_mean).t().matmul((samples - sample_mean)) / n

    assert torch.allclose(mean, sample_mean, atol=1e-2)
    assert torch.allclose(target_covar, sample_covar, atol=1e-2)


@pytest.mark.flaky(max_runs=3, min_passes=1)
def test_sample_normal_diag():
    mean = torch.tensor([1.5, 3.5])
    covar = torch.tensor([0.5, 4.5])
    target_covar = torch.tensor([[0.5, 0.0], [0.0, 4.5]])

    n = 1_000_000
    precisions = cholesky_precision(covar, "diag")
    samples = sample_normal(n, mean, precisions, "diag")

    sample_mean = samples.mean(0)
    sample_covar = (samples - sample_mean).t().matmul((samples - sample_mean)) / n

    assert torch.allclose(mean, sample_mean, atol=1e-2)
    assert torch.allclose(target_covar, sample_covar, atol=1e-2)


@pytest.mark.flaky(max_runs=3, min_passes=1)
def test_sample_normal_full():
    mean = torch.tensor([1.5, 3.5])
    covar = torch.tensor([[4.0, 2.5], [2.5, 2.0]])

    n = 1_000_000
    precisions = cholesky_precision(covar, "tied")
    samples = sample_normal(n, mean, precisions, "full")

    sample_mean = samples.mean(0)
    sample_covar = (samples - sample_mean).t().matmul((samples - sample_mean)) / n

    assert torch.allclose(mean, sample_mean, atol=1e-2)
    assert torch.allclose(covar, sample_covar, atol=1e-2)


# -------------------------------------------------------------------------------------------------


def _assert_log_prob(
    actual: torch.Tensor, x: torch.Tensor, means: torch.Tensor, covars: torch.Tensor
) -> None:
    distribution = MultivariateNormal(means, covars)
    expected = distribution.log_prob(x.unsqueeze(1))
    assert torch.allclose(actual, expected, rtol=1e-3)
