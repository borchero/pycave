import torch
from pycave.bayes.core import CovarianceType
from pycave.bayes.gmm import GaussianMixtureModel, GaussianMixtureModelConfig


def sample_gmm(
    num_datapoints: int, num_features: int, num_components: int, covariance_type: CovarianceType
) -> torch.Tensor:
    config = GaussianMixtureModelConfig(num_components, num_features, covariance_type)
    model = GaussianMixtureModel(config)

    # Means and covariances can simply be scaled
    model.means.mul_(torch.rand(num_components).unsqueeze(-1) * 10).add_(
        torch.rand(num_components).unsqueeze(-1) * 10
    )

    return model.sample(num_datapoints)
