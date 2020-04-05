import torch
import torch.nn as nn
import torch.distributions as dist
import pyblaze.nn as xnn
from pycave.bayes._internal.output import Gaussian
from pycave.bayes._internal.utils import log_responsibilities
from .config import GMMConfig
from .engine import GMMEngine

class GMM(xnn.Estimator, xnn.Configurable, nn.Module):
    """
    The GMM represents a mixture of a fixed number of multivariate gaussian distributions. This
    class may be used to find clusters whenever you expect data to be generated from a (fixed-size)
    set of gaussian distributions.
    """

    __config__ = GMMConfig
    __engine__ = GMMEngine

    def __init__(self, *args, **kwargs):
        """
        Initializes a new GMM either with a given `GMMConfig` or with the config's parameters passed
        as keyword arguments.
        """
        super().__init__(*args, **kwargs)

        self.component_weights = nn.Parameter(torch.empty(self.num_components), requires_grad=False)
        self.gaussian = Gaussian(self.num_components, self.num_features, self.covariance)

        self.reset_parameters()

    def reset_parameters(self, data=None, max_iter=100):
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
        """
        # 1) Gaussian distributions
        labels = self.gaussian.reset_parameters(data, max_iter)

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
        torch.Tensor [1]
            The negative log-likelihood of the data.
        """
        probs = self.gaussian.evaluate(data, log=True)
        log_resp, log_likeli = log_responsibilities(
            probs, self.component_weights, return_log_likelihood=True
        )
        return log_resp.exp(), -log_likeli

    def fit(self, *args, **kwargs):
        """
        Optimizes the HMM's parameters according to some data. This method accepts a multitude of
        parameters which are documented `here <https://bit.ly/39FoKpe>`_. The first parameter of
        this method must be the data. By default, data should be given as PyTorch data loader.
        However, in order to make calling this function easier for simple use cases, you can also
        supply one of the following instead.

        * A single tensor (interpreted as a single batch of datapoints)
        * A list of tensors (interpreted as batches of datapoints)

        Additional parameters are documented below.

        Parameters
        ----------
        eps: float, default: 1e-7
            The minimum per-datapoint difference in the negative log-likelihood to consider a
            model "better", thus indicating convergence.

        Returns
        -------
        pyblaze.nn.History
            A history object with a `neg_log_likelihood` attribute describing the development of
            the negative log-likelihood over the course of training.
        """
        data = self._process_input(args[0])
        return super().fit(data, *args[1:], **kwargs)

    def evaluate(self, *args, **kwargs):
        """
        Returns the negative log-likelihood of the given data according to the model's parameters.
        The data may be given in the same way as for the :meth:`fit` method.

        Returns
        -------
        pyblaze.nn.Evaluation
            An evaluation object where the `neg_log_likelihood` property yields the per-datapoint
            negative log-likelihood.
        """
        data = self._process_input(args[0])
        return super().evaluate(data, *args[1:], **kwargs)

    def predict(self, *args, **kwargs):
        """
        Returns a distribution over components of the GMM indicating the probability that a
        datapoint has been sampled from the respective component. Running argmax over the last
        dimension yields the most likely components for the given data. The data may be given in the
        same way as for the :meth:`fit` method.

        Returns
        -------
        torch.Tensor [N, K]
            The distribution of every datapoint over all components (number of datapoints N,
            number of components K).
        """
        data = self._process_input(args[0])
        return super().predict(data, *args[1:], **kwargs)

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

    def _process_input(self, data):
        if isinstance(data, torch.Tensor):
            return [data]
        return data
