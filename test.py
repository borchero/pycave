import time
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GMM
from pycave.bayes import GaussianMixture
from pycave.bayes.gmm import GaussianMixtureModel, GaussianMixtureModelConfig


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
    chain = GaussianMixture(
        num_components=40,
        covariance_type="spherical",
        # batch_size=10000,
        init_strategy="kmeans",
        trainer_params=dict(
            num_nodes=1,
        ),
    )

    np.random.seed(0)
    torch.manual_seed(0)
    data = _generate_sample_data(100000, 128)

    tic = time.time()
    # chain.fit(data)
    print(f"PyCave took {time.time() - tic} seconds")
    # print(chain.num_iter_)
    # print(chain.nll_)

    # print(chain.score(data))

    # print(chain.model_.means)
    # print(chain.num_iter_)

    tic = time.time()
    m = KMeans(40, n_init=1, algorithm="full")
    m.fit(data.numpy())
    print(f"Sklearn kmeans took {time.time() - tic} seconds")
    print(m.inertia_ / data.size(0))
    print(m.n_iter_)

    m = GMM(40, covariance_type="spherical", init_params="kmeans", verbose=1)
    tic = time.time()
    m.fit(data.numpy())
    print(f"Sklearn took {time.time() - tic} seconds")
    # print(m.means_)
    print(m.n_iter_)
    print(m.score(data.numpy()))
