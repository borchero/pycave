import math
import time
import torch
import torch.nn as nn
import torch.distributions as dist
import bxtorch.nn as xnn
from bxtorch.utils.stdio import ProgressBar
from bxtorch.utils.torch import to_one_hot
from sklearn.cluster import KMeans
from .utils import log_normal, log_responsibilities, max_likeli_means, max_likeli_covars

class GMMConfig(xnn.Config):
    """
    The GMM config can be used to customize the Gaussian mixture model. Values that do not have a
    default value must be passed to the initializer of the GMM.
    """

    num_components: int
    """
    The number of gaussian distributions that make up the GMM.
    """

    num_features: int
    """
    The dimensionality of the gaussian distributions.
    """

    covariance: str = 'diag'
    """
    The type of covariance to use for the gaussian distributions. Must be one of:

    * diag: Diagonal covariance for every component (parameters: `num_features * num_components`).
    * spherical: Spherical covariance for every component (parameters: `num_components`).
    * diag-shared: Shared diagonal covariance for all components (parameters: `num_features`).
    """

    def is_valid(self):
        return (
            self.covariance in ('diag', 'spherical', 'diag-shared')
        )

# pylint: disable=abstract-method
class GMM(xnn.Configurable, xnn.Estimator, nn.Module):
    """
    The GMM represents a mixture of a fixed number of multivariate gaussian distributions. This
    class may be used to find clusters whenever you expect data to be generated from a (fixed-size)
    set of gaussian distributions.
    """

    __config__ = GMMConfig

    # MARK: Initialization
    def __init__(self, *args, **kwargs):
        """
        Initializes a new GMM either with a given `GMMConfig` or with the config's parameters passed
        as keyword arguments.
        """
        super().__init__(*args, **kwargs)

        self.component_weights = nn.Parameter(
            torch.empty(self.num_components), requires_grad=False
        )

        self.means = nn.Parameter(
            torch.empty(self.num_components, self.num_features),
            requires_grad=False
        )

        if self.covariance == 'diag':
            self.covars = nn.Parameter(
                torch.empty(self.num_components, self.num_features),
                requires_grad=False
            )
        elif self.covariance == 'spherical':
            self.covars = nn.Parameter(
                torch.empty(self.num_components), requires_grad=False
            )
        else:
            self.covars = nn.Parameter(
                torch.empty(self.num_features), requires_grad=False
            )

        self.reset_parameters()

    # MARK: Instance Methods
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
        # 1) Means
        if data is not None:
            model = KMeans(
                self.num_components, n_init=1, max_iter=max_iter, n_jobs=-1
            )
            model.fit(data.cpu().numpy())
            # pylint: disable=not-callable
            labels = torch.tensor(model.labels_, dtype=torch.long, device=data.device)
            one_hot_labels = to_one_hot(labels, self.num_components)
            self.means.set_(max_likeli_means(
                data, one_hot_labels, one_hot_labels.sum(0)
            ))
        else:
            labels = None
            self.means.normal_()

        # 2) Covariances
        #    Covariance is estimated via the labels from kmeans if they exist,
        #    otherwise all covariances are set to 1.
        if labels is None:
            self.covars.fill_(1)
        else:
            self.covars.set_(
                max_likeli_covars(
                    data, one_hot_labels, one_hot_labels.sum(0),
                    self.means, self.covariance
                )
            )

        # 3) Components
        if labels is None:
            self.component_weights.uniform_(0, 1)
        else:
            _, counts = torch.unique(labels, return_counts=True)
            self.component_weights.set_(counts.float())

        self.component_weights /= self.component_weights.sum()

    def fit(self, data, max_iter=100, batch_size=None, eps=1e-7, verbose=False):
        """
        Optimizes the GMM's parameters according to the given data. We use (mini-batch)
        expectation-maximization for optimizing the parameters. Consider running `reset_parameters`
        on (a subset of) the data first to achieve better results.

        Parameters
        ----------
        data: torch.Tensor [N, D]
            The data to train the model on. The data does not need to be passed to the GPU when
            training on mini-batches (i.e. `batch_size` is given). If the model itself resides on
            the GPU, it will take care of moving the required data to the GPU step by step,
            enabling training on data that is larger than the memory of the GPU. In case the model
            is trained without mini-batches, the data needs to be on the same device as the model.
        max_iter: int, default: 100
            The maximum number of iterations to run the EM-algorithm for.
        batch_size: int, default: None
            The batch size to use for training. If `None` is given, we do not uses batches.
        eps: float, default: 1e-7
            The difference in the log-likelihood (per datapoint) after which to stop the
            EM-algorithm before running `max_iter` iterations. The default value works well for
            reasonably large datasets (~10 million datapoins) but might be set to an even smaller
            value for even larger datasets.
        verbose: bool, default: False
            Whether to show a progress bar during training.

        Returns
        -------
        bxtorch.nn.History
            A history object containing the negative log-likelihoods over the course of the
            EM-training.
        """
        history = []
        tic = time.time()

        # 1) Initialize batch weights
        if batch_size is None:
            num_batches = 1
        else:
            num_batches = int(math.ceil(data.size(0) / batch_size))
        weights = torch.ones(num_batches)

        # 1.1) Adjust weight of last batch
        if batch_size is not None:
            last_batch_size = data.size(0) - batch_size * (weights.size(0) - 1)
            weights[-1] = last_batch_size / batch_size

        # 1.2) Normalize weights
        weights = weights / weights.sum()

        # 1.3) Setup early stopping
        best_neg_log_likeli = float('inf')

        # 2) Run training
        for _ in ProgressBar(max_iter, verbose=verbose):
            if num_batches == 1:
                updates = self._em_full(data)
            else:
                updates = self._em_batch(data, batch_size, weights)

            # 2.1) Update parameters
            self.component_weights.set_(updates['component_weights'])
            self.means.set_(updates['means'])
            self.covars.set_(updates['covars'])

            # 2.2) Update history
            neg_log_likeli = updates['neg_log_likelihood']
            history.append({
                'neg_log_likelihood': neg_log_likeli
            })

            # 2.3) Check for early stopping
            if best_neg_log_likeli - neg_log_likeli < eps:
                break
            best_neg_log_likeli = neg_log_likeli

        return xnn.History(time.time() - tic, history)

    def evaluate(self, data, return_responsibilities=False):
        """
        Computes the negative log-likelihood of the given data (normalized by the number of
        datapoints).

        Parameters
        ----------
        data: torch.Tensor [N, D]
            The data to compute the log-likelihood for.
        return_responsibilities: bool, default: False
            Whether to return the responsibilities for the datapoints and each component.

        Returns
        -------
        float
            The per-datapoint negative log-likelihood.
        torch.Tensor [N, K]
            The responsibilities when `return_responsibilities` is set.
        """
        probs = log_normal(data, self.means, self.covars)
        log_resp, log_likeli = log_responsibilities(
            probs, self.component_weights, return_log_likelihood=True
        )
        neg_log_likeli = -log_likeli / data.size(0)

        if return_responsibilities:
            return neg_log_likeli, log_resp.exp()
        return neg_log_likeli

    def predict(self, data):
        """
        Predicts the most likely components for the given data.

        Parameters
        ----------
        data: torch.Tensor [..., D]
            The data for which to compute the most likely components that they were generated from.
            The size is arbitrary, only the last dimension must be the same as the dimensionality
            of the gaussian distributions.

        Returns
        -------
        torch.Tensor [...]
            The indices of the most likely components. The shape is the same as of the given data,
            except for the last dimension.
        """
        size = data.size()

        log_probs = log_normal(data.view(-1, size[-1]), self.means, self.covars)
        priors = self.component_weights.view(1, -1)
        components = (log_probs * priors).argmax(-1)

        return components.view(*size[:-1])

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
        ret = self._sample_from_components(components)

        if return_components:
            return ret, components
        return ret

    # MARK: Private Methods
    def _sample_components(self, num_samples):
        generator = dist.Categorical(self.component_weights)
        return generator.sample((num_samples,))

    def _sample_from_components(self, components):
        samples = torch.empty(
            components.size(0), self.num_features,
            device=components.device, dtype=torch.float
        )

        unique_components, component_counts = torch.unique(
            components, return_counts=True
        )

        for i in range(unique_components.size(0)):
            c = unique_components[i]
            samples[components == c] = \
                dist.MultivariateNormal(
                    self.means[c], self._get_full_covariance_matrix(c)
                ).sample((component_counts[i],))

        return samples

    def _get_full_covariance_matrix(self, i):
        if self.covariance == 'diag':
            return torch.diag(self.covars[i])
        if self.covariance == 'diag-shared':
            return torch.diag(self.covars)
        return self.covars[i] * torch.diag(torch.ones(self.num_features))

    def _estimate(self, data, batch=False):
        # 1) Compute responsibilities for components and data likelihood
        neg_log_likeli, responsibilities = self.evaluate(
            data, return_responsibilities=True
        )

        # 2) Compute maximum likelihood parameters
        comp_sums = responsibilities.sum(0)
        comp_sums += torch.finfo(torch.float).eps

        # 2.1) Component weights
        component_weights = comp_sums / comp_sums.sum()

        # 2.2) Means
        means = max_likeli_means(
            data, responsibilities, comp_sums if not batch else None
        )

        if batch:
            # In case of batching, we need to compute the covariance at a later point.
            return {
                'component_weights': component_weights,
                'component_sums': comp_sums,
                'means': means,
                'neg_log_likelihood': neg_log_likeli
            }

        # 2.3) Covariances
        covars = max_likeli_covars(
            data, responsibilities, comp_sums, means, self.covariance
        )

        return {
            'component_weights': component_weights,
            'means': means,
            'covars': covars,
            'neg_log_likelihood': neg_log_likeli
        }

    def _em_full(self, data):
        estimates = self._estimate(data)

        nll = (estimates['neg_log_likelihood'] / data.size(0)).item()
        return {
            'component_weights': estimates['component_weights'],
            'means': estimates['means'],
            'covars': estimates['covars'],
            'neg_log_likelihood': nll
        }

    def _em_batch(self, data, batch_size, weights):
        # 1) Setup
        num_batches = weights.size(0)
        device = self.component_weights.device

        # 1.1) Helper functions
        def get_batch(i):
            if i == num_batches - 1:
                batch = data[i*batch_size:]
            else:
                batch = data[i*batch_size:(i+1)*batch_size]

            # Move to device to enable data on CPU and moving to GPU only during computations
            return batch.to(device)

        # 2) Initialize containers to aggregate information from different batches.
        component_weights = torch.zeros_like(self.component_weights)
        component_sums = torch.zeros(self.num_components, device=device)
        means = torch.zeros_like(self.means)
        neg_log_likeli = 0

        # 3) Estimate likelihood, compute component weights and means
        for i in range(num_batches):
            # 3.1) Get batch
            batch = get_batch(i)

            # 3.2) Estimate
            estimates = self._estimate(batch, batch=True)

            # 3.3) Aggregate batch results
            component_weights += estimates['component_weights'] * weights[i]
            component_sums += estimates['component_sums'] * weights[i]
            means += estimates['means'] * weights[i]
            neg_log_likeli += (estimates['neg_log_likelihood'] / data.size(0)) * weights[i]

        # 3.4) Update means
        means = means / component_sums.unsqueeze(1)

        # 4) Now, given the overall means, we compute the covariances. For this, we naturally need
        # the responsibilities. Although we computed them earlier, we choose to recompute them as,
        # otherwise, memory consumption would be too high and there wouldn't be a reason to perform
        # batching in the first place.

        # 4.1) Initialize containers - here, we can reuse `component_sums` from above as the
        # responsibilities do not change.
        K = self.num_components
        D = self.num_features
        covars_x_sq = torch.zeros(K, D, device=device)
        covars_xm_wo_means = torch.zeros(K, D, device=device)

        covars = torch.zeros_like(self.covars)

        for i in range(num_batches):
            # 4.1) Get batch
            batch = get_batch(i)

            # 4.2) Compute responsibilities
            resp = self._responsibilities(batch)

            # 4.3) Now we actually compute the covariance. Due to the odd division by the
            # component_sums, we cannot use the functions from `.utils` and use an adjusted version
            # from the code there.
            if self.covariance in ('diag', 'spherical'):
                covars_x_sq += torch.matmul(resp.t(), batch * batch) * weights[i]
                covars_xm_wo_means += torch.matmul(resp.t(), batch) * weights[i]
            else:
                pass

        # 4.4) After having processed all batches, we can aggregate the results to compute the
        # actual covariance
        if self.covariance in ('diag', 'spherical'):
            denom = component_sums.unsqueeze(1)
            x_sq = covars_x_sq / denom
            m_sq = means * means
            xm = means * covars_xm_wo_means / denom
            covars = x_sq - 2 * xm + m_sq + 1e-6 # regularization
            if self.covariance == 'spherical':
                covars = covars.mean(1)
        else:
            pass

        return {
            'component_weights': component_weights,
            'means': means,
            'covars': covars,
            'neg_log_likelihood': neg_log_likeli
        }

    def _responsibilities(self, data):
        probs = log_normal(data, self.means, self.covars)
        log_resp = log_responsibilities(probs, self.component_weights)
        return log_resp.exp()
