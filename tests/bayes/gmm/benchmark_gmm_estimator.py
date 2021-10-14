import pytest
import torch
from pytest_benchmark.fixture import BenchmarkFixture  # type: ignore
from sklearn.mixture import GaussianMixture as SklearnGaussianMixture  # type: ignore
from pycave.bayes import GaussianMixture
from pycave.bayes.gmm_ import GaussianMixtureModel, GaussianMixtureModelConfig


def _generate_sample_data(num_datapoints: int, num_features: int) -> torch.Tensor:
    num_components = 4

    # Initialize model
    mixture = GaussianMixtureModel(
        GaussianMixtureModelConfig(
            num_components=num_components, num_features=num_features, covariance_type="spherical"
        )
    )

    # Set custom parameters
    mixture.component_probs.copy_(torch.ones(num_components) / num_components)
    mixture.means.copy_(
        torch.stack(
            [
                torch.zeros(num_features) - 2,
                torch.zeros(num_features) - 1,
                torch.zeros(num_features) + 1,
                torch.zeros(num_features) + 2,
            ]
        )
    )
    mixture.precisions_cholesky.copy_(torch.zeros(num_components) + 3)

    # Return samples
    return mixture.sample(num_datapoints)


# -------------------------------------------------------------------------------------------------


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
