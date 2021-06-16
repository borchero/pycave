import torch
import torch.nn as nn
import torch.distributions as dist
import pyblaze.nn as xnn
from pycave.bayes._internal.output import Gaussian
from pycave.bayes._internal.utils import log_responsibilities
from .engine import GMMEngine

class GMM(xnn.Estimator, nn.Module):
    """
    The GMM represents a mixture of a fixed number of multivariate gaussian distributions. This
    class may be used to find clusters whenever you expect data to be generated from a (fixed-size)
    set of gaussian distributions.

    In addition to the methods documented below, the GMM provides the following methods as provided
    by the `estimator mixin <https://bit.ly/2wiUB1i>`_.

    `fit(...)`
        Optimizes the model's parameters.

    `evaluate(...)`
        Computes the per-datapoint negative log-likelihood of the given data.

    `predict(...)`
        Computes the probability distribution over components for each datapoint. An argmax over
        the component dimension thus yields the most likely states.

    The parameters that may be passed to the functions can be derived from the
    `engine documentation <https://bit.ly/3bYHhOV>`_. The data needs, however, not be passed as a
    PyTorch data loader but all methods also accept the following instead:

    * A single tensor (interpreted as a single batch of datapoints)
    * A list of tensors (interpreted as batches of datapoints)

    Additionally, the methods allow the following keyword arguments:

    `fit(...)`
        eps: float, default: 0.01
            The minimum per-datapoint difference in the negative log-likelihood to consider a
            model "better", thus indicating convergence.
        reg: float, default: 1e-6
            A non-negative regularization term to be added to the diagonal of the covariance matrix
            to ensure that it is positive. If your data contains datapoints which are very close
            together (i.e. "singleton datapoints"), you may need to increase that regularization
            factor.

    `evaluate(...)`
        reduction: str, default: 'mean'
            The reduction performed for the negative log-likelihood as for common PyTorch metrics.
            Must be one of ['mean', 'sum', 'none'].
    """

    @property
    def engine(self):
        return GMMEngine(self)

    def __init__(self, num_components, num_features, covariance='diag'):
        """
        Initializes a new GMM.

        Parameters
        ----------
        num_components: int
            The number of gaussian distributions that make up the GMM.
        num_features: int
            The dimensionality of the gaussian distributions.
        covariance: str, default: 'diag'
            The type of covariance to use for the gaussian distributions. Must be one of:

            * diag: Diagonal covariance for every component (parameters: `num_features *
                num_components`).
            * spherical: Spherical covariance for every component (parameters: `num_components`).
            * diag-shared: Shared diagonal covariance for all components (parameters:
                `num_features`).
        """
        super().__init__()

        if covariance not in ('diag', 'spherical', 'diag-shared'):
            raise ValueError(f"Invalid covariance type '{covariance}'.")

        self.num_components = num_components
        self.num_features = num_features
        self.covariance = covariance

        self.component_weights = nn.Parameter(torch.empty(self.num_components), requires_grad=False)
        self.gaussian = Gaussian(self.num_components, self.num_features, self.covariance)

        self.reset_parameters()

    def reset_parameters(self, data=None, max_iter=100, reg=1e-6):
        """
        Initializes the parameters of the GMM, optionally based on some data. If no data is given,
        means are initialized randomly from a gaussian distribution, unit covariances are used and
        prior probabilities are assigned randomly using a uniform distribution.

        Parameters
        ----------
        data: torch.Tensor [N, D], default: None
            An optional set of datapoints to initialize the means and covariances of the gaussian
            distributions from. K-Means will be run to find the means and the datapoints belonging
            to a respective cluster are used to estimate the covariance. Note that the given data
            may be a (small) subset on the actual data that the GMM should be fitted on.
        max_iter: int, default: 100
            If data is given and K-Means is run, this defines the maximum number of iterations to
            run K-Means for.
        reg: float, default: 1e-6
            A non-negative regularization term to be added to the diagonal of the covariance matrix
            to ensure that it is positive. If your data contains datapoints which are very close
            together (i.e. "singleton datapoints"), you may need to increase that regularization
            factor. This parameter is ignored if no data is provided.
        """
        # 1) Gaussian distributions
        labels = self.gaussian.reset_parameters(data, max_iter, reg=reg)

        # 2) Components
        if labels is None:
            self.component_weights.uniform_(0, 1)
        else:
            _, counts = torch.unique(labels, return_counts=True)
            self.component_weights.set_(counts.float())

        self.component_weights /= self.component_weights.sum()

    def forward(self, data):
        """
        Computes the distribution over components of all datapoints as well as the negative
        log-likelihood of the given data.

        Parameters
        ----------
        data: torch.Tensor [N, D]
            The data to perform computations for (number of datapoints N, dimensionality D).

        Returns
        -------
        torch.Tensor [N, K]
            The responsibilities for each datapoint and component (number of components K).
        torch.Tensor [N]
            The negative log-likelihood for all data samples.
        """
        probs = self.gaussian.evaluate(data, log=True)
        log_resp, log_likeli = log_responsibilities(
            probs, self.component_weights, return_log_likelihood=True
        )
        return log_resp.exp(), -log_likeli.squeeze(-1)

    def sample(self, n, return_components=False):
        """
        Samples a given number of samples from the GMM.

        Parameters
        ----------
        n: int
            The number of samples to generate.
        return_components: bool, default: False
            Whether to return the indices of the components from which the samples were obtained.

        Returns
        -------
        torch.Tensor [N, D]
            The samples with dimensionality D.
        torch.Tensor [N]
            Optionally, the indices of the components corresponding to the returned samples.
        """
        # 1) Sample components
        components = self._sample_components(n)

        # 2) Sample from the components
        ret = self.gaussian.sample(components)

        if return_components:
            return ret, components
        return ret

    def _sample_components(self, num_samples):
        generator = dist.Categorical(self.component_weights)
        return generator.sample((num_samples,))

    def prepare_input(self, data):
        if isinstance(data, torch.Tensor):
            return [data]
        return data
