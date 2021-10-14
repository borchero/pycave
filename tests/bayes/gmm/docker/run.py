import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import CSVLogger
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


if __name__ == "__main__":
    logger = CSVLogger("lightning_logs")
    trainer = pl.Trainer(accelerator="ddp_cpu", num_nodes=2, logger=logger)
    chain = GaussianMixture(num_components=4, covariance_type="spherical", trainer=trainer)
    data = _generate_sample_data(100000, 8)
    chain.fit(data)
    print(chain.num_iter_)
    print(chain.nll_)

    import os

    print(os.listdir("lightning_logs/default/version_6"))
    with open("lightning_logs/default/version_6/metrics.csv") as f:
        print(f.read())
