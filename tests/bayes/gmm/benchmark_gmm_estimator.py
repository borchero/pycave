# pylint: disable=missing-function-docstring
import pytest
import torch
from pytest_benchmark.fixture import BenchmarkFixture  # type: ignore
from sklearn.mixture import GaussianMixture as SklearnGaussianMixture  # type: ignore
from pycave.bayes import GaussianMixture
from pycave.bayes.gmm import GaussianMixtureModel, GaussianMixtureModelConfig


@pytest.mark.parametrize(
    "num_datapoints,num_features", zip([10_000, 100_000, 1_000_000], [8, 64, 512])
)
def test_fit(benchmark: BenchmarkFixture, num_datapoints: int, num_features: int):
    torch.manual_seed(42)
    data = _generate_sample_data(num_datapoints, num_features)

    mixture = GaussianMixture(num_components=4, covariance_type="diag")
    benchmark(mixture.fit, data)

    print(mixture.num_iter_, mixture.converged_, mixture.nll_)
    assert False


@pytest.mark.parametrize(
    "num_datapoints,num_features", zip([10_000, 100_000, 1_000_000], [8, 64, 512])
)
def test_sklearn_fit(benchmark: BenchmarkFixture, num_datapoints: int, num_features: int):
    torch.manual_seed(42)
    data = _generate_sample_data(num_datapoints, num_features).numpy()

    mixture = SklearnGaussianMixture(n_components=4, covariance_type="diag", init_params="random")
    benchmark(mixture.fit, data)

    print(mixture.n_iter_, mixture.converged_, -mixture.lower_bound_)
    assert False
