from __future__ import annotations
import logging
from typing import Any, cast, Dict, List, Optional, Tuple
import torch
from pycave.bayes.core import CovarianceType
from pycave.clustering import KMeans
from pycave.core.estimator import Estimator
from pycave.data import TabularData
from .lightning_module import GaussianMixtureLightningModule
from .model import GaussianMixtureModel, GaussianMixtureModelConfig
from .types import GaussianMixtureInitStrategy

logger = logging.getLogger(__name__)


class GaussianMixture(Estimator[GaussianMixtureModel]):
    """
    Probabilistic model assuming that data is generated from a mixture of Gaussians.

    A Gaussian mixture can be used to learn the latent Gaussian distributions (i.e. components)
    from which data is sampled. More information is available
    `on Wikipedia <https://en.wikipedia.org/wiki/Mixture_model>`_.

    See also:
        .. currentmodule:: pycave.bayes.gmm
        .. autosummary::
            :nosignatures:
            :template: classes/pytorch_module.rst

            GaussianMixtureModel
            GaussianMixtureModelConfig
    """

    def __init__(
        self,
        num_components: int = 1,
        covariance_type: CovarianceType = "diag",
        init_strategy: GaussianMixtureInitStrategy = "kmeans",
        convergence_tolerance: float = 1e-3,
        reg_covar: float = 1e-6,
        batch_size: Optional[int] = None,
        num_workers: int = 0,
        trainer_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            num_components: The number of components in the GMM. The dimensionality of each
                component is automatically inferred from the data.
            covariance_type: The type of covariance to assume for all Gaussian components.
            init_strategy: The strategy for initializing component means and covariances.
            batch_size: The batch size to use when fitting the model. If not provided, the full
                data will be used as a single batch. Set this if the full data does not fit into
                memory.
            num_workers: The number of workers to use for loading the data.
            trainer_params: Initialization parameters to use when initializing a PyTorch Lightning
                trainer. This estimator sets the following overridable defaults:

                - ``max_epochs=100``
                - ``checkpoint_callback=False``
                - ``log_every_n_steps=1``
                - ``weights_summary=None``

        Note:
            The GMM is trained via the EM algorithm. When training via mini-batches, epochs
            alternately compute the updated means (as well as component priors) and covariances,
            respectively. If training is stopped after an odd number of epochs, only the GMM's
            means and component prior will have been updated while the covariances are outdated.
            Thus, training for and odd number of epochs is discouraged.
        """
        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            default_params=dict(
                max_epochs=100,
                checkpoint_callback=False,
                log_every_n_steps=1,
                weights_summary=None,
            ),
            user_params=trainer_params,
        )

        self.num_components = num_components
        self.covariance_type = covariance_type
        self.init_strategy: GaussianMixtureInitStrategy = init_strategy
        self.convergence_tolerance = convergence_tolerance
        self.reg_covar = reg_covar

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.converged_: bool
        self.num_iter_: int
        self.nll_: float

    def fit(self, data: TabularData) -> GaussianMixture:
        """
        Fits the Gaussian mixture on the provided data, estimating component priors, means and
        covariances.

        Args:
            data: The tabular data to fit on. The dimensionality of the Gaussian mixture is
                automatically inferred from this data.

        Returns:
            The fitted Gaussian mixture.
        """
        self._init_trainer()

        # Initialize model
        num_features = len(data[0])
        config = GaussianMixtureModelConfig(
            num_components=self.num_components,
            num_features=num_features,
            covariance_type=self.covariance_type,  # type: ignore
        )
        self._model = GaussianMixtureModel(config)

        # Initialize the means if required
        if self.init_strategy == "kmeans":
            logger.info("Running k-means initialization.")
            estimator = KMeans(
                self.num_components,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                trainer_params=self._trainerparams_user,
            ).fit(data)
            self.model_.means.copy_(estimator.model_.centroids)

        # Set up data loader
        loader = self._init_data_loader(data, for_training=True)
        batch_training = self._uses_batch_training(loader)

        # Fit model
        module = GaussianMixtureLightningModule(
            self.model_,
            tol=self.convergence_tolerance,
            reg_covar=self.reg_covar,
            init_strategy=self.init_strategy,
            batch_training=batch_training,
        )
        self._trainer.fit(module, loader)

        # Assign convergence properties
        self.num_iter_ = module.current_epoch + 1
        if batch_training:
            # For batch training, the actual number of iterations is lower. For random
            # initialization, this should be an even number, for kmeans initialization, it should
            # be an odd number.
            self.num_iter_ //= 2
        # For random initialization, one optimization cycle is used
        self.num_iter_ -= 1

        self.converged_ = module.current_epoch + 1 < cast(int, self._trainer.max_epochs)
        self.nll_ = cast(float, self._trainer.callback_metrics["nll"].item())
        return self

    def sample(self, num_datapoints: int) -> torch.Tensor:
        """
        Samples datapoints from the fitted Gaussian mixture.

        Args:
            num_datapoints: The number of datapoints to sample.

        Returns:
            A tensor of shape ``[num_datapoints, dim]`` providing the samples.
        """
        return self.model_.sample(num_datapoints)

    def score(self, data: TabularData) -> float:
        """
        Computes the average negative log-likelihood (NLL) of the provided datapoints.

        Args:
            data: The datapoints for which to evaluate the NLL.

        Returns:
            The average NLL of all datapoints.

        Note:
            See :meth:`score_samples` to obtain NLL values for individual datapoints.
        """
        result = self._trainer.test(
            GaussianMixtureLightningModule(self.model_),
            self._init_data_loader(data, for_training=False),
            verbose=False,
        )
        return result[0]["nll"]

    def score_samples(self, data: TabularData) -> torch.Tensor:
        """
        Computes the negative log-likelihood (NLL) of each of the provided datapoints.

        Args:
            data: The datapoints for which to compute the NLL.

        Returns:
            A tensor of shape ``[num_datapoints]`` with the NLL for each datapoint.
        """
        result = self._trainer.predict(
            GaussianMixtureLightningModule(self.model_),
            self._init_data_loader(data, for_training=False),
        )
        return torch.stack([x[1] for x in cast(List[Tuple[torch.Tensor, torch.Tensor]], result)])

    def predict(self, data: TabularData) -> torch.Tensor:
        """
        Computes the most likely components for each of the provided datapoints.

        Args:
            data: The datapoints for which to obtain the most likely components.

        Returns:
            A tensor of shape ``[num_datapoints]`` with the indices of the most likely components.

        Note:
            Use :meth:`predict_proba` to obtain probabilities for each component instead of the
            most likely component only.
        """
        return self.predict_proba(data).argmax(-1)

    def predict_proba(self, data: TabularData) -> torch.Tensor:
        """
        Computes a distribution over the components for each of the provided datapoints.

        Args:
            data: The datapoints for which to compute the component assignment probabilities.

        Returns:
            A tensor of shape ``[num_datapoints, num_components]`` with the assignment
            probabilities for each component and datapoint. Note that each row of the vector sums
            to 1, i.e. the returned tensor provides a proper distribution over the components for
            each datapoint.
        """
        result = self._trainer.predict(
            GaussianMixtureLightningModule(self.model_),
            self._init_data_loader(data, for_training=False),
        )
        return torch.cat([x[0] for x in cast(List[Tuple[torch.Tensor, torch.Tensor]], result)])
