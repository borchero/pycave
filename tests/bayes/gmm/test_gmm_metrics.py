# pylint: disable=protected-access,missing-function-docstring
from typing import Any, Callable
import numpy as np
import sklearn.mixture._gaussian_mixture as skgmm  # type: ignore
import torch
from pycave.bayes.core import CovarianceType
from pycave.bayes.gmm.metrics import CovarianceAggregator, MeanAggregator, PriorAggregator


def test_prior_aggregator():
    aggregator = PriorAggregator(3)
    aggregator.reset()

    # Step 1: single batch
    responsibilities1 = torch.tensor([[0.3, 0.3, 0.4], [0.8, 0.1, 0.1], [0.4, 0.5, 0.1]])
    actual = aggregator.forward(responsibilities1)
    expected = torch.tensor([0.5, 0.3, 0.2])
    assert torch.allclose(actual, expected)

    # Step 2: batch aggregation
    responsibilities2 = torch.tensor([[0.7, 0.2, 0.1], [0.5, 0.4, 0.1]])
    aggregator.update(responsibilities2)
    actual = aggregator.compute()
    expected = torch.tensor([0.54, 0.3, 0.16])
    assert torch.allclose(actual, expected)


def test_mean_aggregator():
    aggregator = MeanAggregator(3, 2)
    aggregator.reset()

    # Step 1: single batch
    data1 = torch.tensor([[5.0, 2.0], [3.0, 4.0], [1.0, 0.0]])
    responsibilities1 = torch.tensor([[0.3, 0.3, 0.4], [0.8, 0.1, 0.1], [0.4, 0.5, 0.1]])
    actual = aggregator.forward(data1, responsibilities1)
    expected = torch.tensor([[2.8667, 2.5333], [2.5556, 1.1111], [4.0, 2.0]])
    assert torch.allclose(actual, expected, atol=1e-4)

    # Step 2: batch aggregation
    data2 = torch.tensor([[8.0, 2.5], [1.5, 4.0]])
    responsibilities2 = torch.tensor([[0.7, 0.2, 0.1], [0.5, 0.4, 0.1]])
    aggregator.update(data2, responsibilities2)
    actual = aggregator.compute()
    expected = torch.tensor([[3.9444, 2.7963], [3.0, 2.0667], [4.1875, 2.3125]])
    assert torch.allclose(actual, expected, atol=1e-4)


def test_covariance_aggregator_spherical():
    _test_covariance("spherical", skgmm._estimate_gaussian_covariances_spherical)  # type: ignore


def test_covariance_aggregator_diag():
    _test_covariance("diag", skgmm._estimate_gaussian_covariances_diag)  # type: ignore


def test_covariance_aggregator_tied():
    _test_covariance("tied", skgmm._estimate_gaussian_covariances_tied)  # type: ignore


def test_covariance_aggregator_full():
    _test_covariance("full", skgmm._estimate_gaussian_covariances_full)  # type: ignore


def _test_covariance(
    covariance_type: CovarianceType,
    sk_aggregator: Callable[[Any, Any, Any, Any, Any], Any],
):
    reg = 1e-5
    aggregator = CovarianceAggregator(3, 2, covariance_type, reg=reg)
    aggregator.reset()
    means = torch.tensor([[3.0, 2.5], [2.5, 1.0], [4.0, 2.0]])

    # Step 1: single batch
    data1 = torch.tensor([[5.0, 2.0], [3.0, 4.0], [1.0, 0.0]])
    responsibilities1 = torch.tensor([[0.3, 0.3, 0.4], [0.8, 0.1, 0.1], [0.4, 0.5, 0.1]])
    actual = aggregator.forward(data1, responsibilities1, means)
    expected = sk_aggregator(  # type: ignore
        responsibilities1.numpy(),
        data1.numpy(),
        responsibilities1.sum(0).numpy(),
        means.numpy(),
        reg,
    ).astype(np.float32)
    assert torch.allclose(actual, torch.from_numpy(expected))

    # Step 2: batch aggregation
    data2 = torch.tensor([[8.0, 2.5], [1.5, 4.0]])
    responsibilities2 = torch.tensor([[0.7, 0.2, 0.1], [0.5, 0.4, 0.1]])
    aggregator.update(data2, responsibilities2, means)
    actual = aggregator.compute()
    expected = sk_aggregator(  # type: ignore
        torch.cat([responsibilities1, responsibilities2]).numpy(),
        torch.cat([data1, data2]).numpy(),
        (responsibilities1.sum(0) + responsibilities2.sum(0)).numpy(),
        means.numpy(),
        reg,
    ).astype(np.float32)
    assert torch.allclose(actual, torch.from_numpy(expected))
