from dataclasses import dataclass
from typing import Tuple
import numpy as np
import torch
from lightkit.nn import Configurable
from torch import jit, nn
from pycave.bayes.core import covariance_shape, CovarianceType
from pycave.bayes.core._jit import jit_log_normal, jit_sample_normal


@dataclass
class GaussianMixtureModelConfig:
    """
    Configuration class for a Gaussian mixture model.

    See also:
        :class:`GaussianMixtureModel`
    """

    #: The number of components in the GMM.
    num_components: int
    #: The number of features for the GMM's components.
    num_features: int
    #: The type of covariance to use for the components.
    covariance_type: CovarianceType


class GaussianMixtureModel(Configurable[GaussianMixtureModelConfig], nn.Module):
    """
    PyTorch module for a Gaussian mixture model. Covariances are represented via their Cholesky
    decomposition for computational efficiency. The model does not have trainable parameters.
    """

    #: The probabilities of each component, buffer of shape ``[num_components]``.
    component_probs: torch.Tensor
    #: The means of each component, buffer of shape ``[num_components, num_features]``.
    means: torch.Tensor
    #: The precision matrices for the components' covariances, buffer with a shape dependent
    #: on the covariance type, see :class:`CovarianceType`.
    precisions_cholesky: torch.Tensor

    def __init__(self, config: GaussianMixtureModelConfig):
        """
        Args:
            config: The configuration to use for initializing the module's buffers.
        """
        super().__init__(config)

        self.covariance_type = config.covariance_type

        self.register_buffer("component_probs", torch.empty(config.num_components))
        self.register_buffer("means", torch.empty(config.num_components, config.num_features))

        shape = covariance_shape(
            config.num_components, config.num_features, config.covariance_type
        )
        self.register_buffer("precisions_cholesky", torch.empty(shape))

        self.reset_parameters()

    @jit.unused
    def reset_parameters(self) -> None:
        """
        Resets the parameters of the GMM.

        - Component probabilities are initialized via uniform sampling and normalization.
        - Means are initialized randomly from a Standard Normal.
        - Cholesky precisions are initialized randomly based on the covariance type. For all
          covariance types, it is based on uniform sampling.
        """
        nn.init.uniform_(self.component_probs)
        self.component_probs.div_(self.component_probs.sum())

        nn.init.normal_(self.means)

        nn.init.uniform_(self.precisions_cholesky)
        if self.covariance_type == "full":
            self.precisions_cholesky.copy_(
                self.precisions_cholesky.bmm(self.precisions_cholesky.transpose(-1, -2))
            )
        elif self.covariance_type == "tied":
            self.precisions_cholesky.copy_(
                self.precisions_cholesky.mm(self.precisions_cholesky.t())
            )

    def forward(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the log-probability of observing each of the provided datapoints for each of the
        GMM's components.

        Args:
            data: A tensor of shape ``[num_datapoints, num_features]`` for which to compute the
                log-probabilities.

        Returns:
            - A tensor of shape ``[num_datapoints, num_components]`` with the log-responsibilities
              for each datapoint and components. These are the logits of the Categorical
              distribution over the parameters.
            - A tensor of shape ``[num_datapoints]`` with the log-likelihood of each datapoint.
        """
        log_probabilities = jit_log_normal(
            data, self.means, self.precisions_cholesky, self.covariance_type
        )
        log_responsibilities = log_probabilities + self.component_probs.log()
        log_prob = log_responsibilities.logsumexp(1, keepdim=True)
        return log_responsibilities - log_prob, log_prob.squeeze(1)

    def sample(self, num_datapoints: int) -> torch.Tensor:
        """
        Samples the provided number of datapoints from the GMM.

        Args:
            num_datapoints: The number of datapoints to sample.

        Returns:
            A tensor of shape ``[num_datapoints, num_features]`` with the random samples.

        Attention:
            This method does not automatically perform batching. If you need to sample many
            datapoints, call this method multiple times.
        """
        # First, we sample counts for each
        component_counts = np.random.multinomial(num_datapoints, self.component_probs.numpy())

        # Then, we generate datapoints for each components
        result = []
        for i, count in enumerate(component_counts):
            sample = jit_sample_normal(
                count.item(),
                self.means[i],
                self._get_component_precision(i),
                self.covariance_type,
            )
            result.append(sample)

        return torch.cat(result, dim=0)

    def _get_component_precision(self, component: int) -> torch.Tensor:
        if self.covariance_type == "tied":
            return self.precisions_cholesky
        return self.precisions_cholesky[component]
