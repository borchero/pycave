from __future__ import annotations
import logging
from typing import Any, cast, Dict, List, Optional, Tuple
import torch
from pycave.bayes.core import CovarianceType
from pycave.clustering import KMeans
from pycave.core import Estimator, PredictorMixin
from pycave.data import TabularData
from .lightning_module import GaussianMixtureLightningModule
from .model import GaussianMixtureModel, GaussianMixtureModelConfig
from .types import GaussianMixtureInitStrategy

logger = logging.getLogger(__name__)


class GaussianMixture(Estimator[GaussianMixtureModel], PredictorMixin[TabularData, torch.Tensor]):
    """
    Probabilistic model assuming that data is generated from a mixture of Gaussians. The mixture is
    assumed to be composed of a fixed number of components with individual means and covariances.
    More information on Gaussian mixture models (GMMs) is available on
    `Wikipedia <https://en.wikipedia.org/wiki/Mixture_model>`_.

    See also:
        .. currentmodule:: pycave.bayes.gmm
        .. autosummary::
            :nosignatures:
            :template: classes/pytorch_module.rst

            GaussianMixtureModel
            GaussianMixtureModelConfig
    """

    #: A boolean indicating whether the model converged during training.
    converged_: bool
    #: The number of iterations the model was fitted for, excluding initialization.
    num_iter_: int
    #: The average per-datapoint negative log-likelihood at the last training step.
    nll_: float

    def __init__(
        self,
        num_components: int = 1,
        *,
        covariance_type: CovarianceType = "diag",
        init_strategy: GaussianMixtureInitStrategy = "kmeans",
        convergence_tolerance: float = 1e-3,
        covariance_regularization: float = 1e-6,
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
            convergence_tolerance: The change in the per-datapoint negative log-likelihood which
                implies that training has converged.
            covariance_regularization: A small value which is added to the diagonal of the
                covariance matrix to ensure that it is positive semi-definite.
            batch_size: The batch size to use when fitting the model. If not provided, the full
                data will be used as a single batch. Set this if the full data does not fit into
                memory.
            num_workers: The number of workers to use for loading the data.
            trainer_params: Initialization parameters to use when initializing a PyTorch Lightning
                trainer. By default, it disables various stdout logs unless PyCave is configured to
                do verbose logging. Checkpointing and logging are disabled regardless of the log
                level. This estimator further sets the following overridable defaults:

                - ``max_epochs=100``

        Note:
            The number of epochs passed to the initializer only define the number of optimization
            epochs. Prior to that, a single initialization epoch is run.

        Note:
            For batch training, the number of epochs run (i.e. the number of passes through the
            data), does not align with the number of epochs passed to the optimizer. This is
            because the EM-algorithm needs to be split up across two epochs. The actual number of
            minimum/maximum epochs is, thus, doubled.
        """
        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            default_params=dict(
                max_epochs=100,
            ),
            user_params=trainer_params,
        )

        self.num_components = num_components
        self.covariance_type = covariance_type
        self.init_strategy = init_strategy
        self.convergence_tolerance = convergence_tolerance
        self.covariance_regularization = covariance_regularization

        self.batch_size = batch_size
        self.num_workers = num_workers

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
        # Init the trainer
        batch_training = self._uses_batch_training(data)  # type: ignore
        self._init_trainer(
            updated_params=dict(
                max_epochs=(
                    self.trainer_params["max_epochs"] * (int(batch_training) + 1)
                    + int(self.init_strategy != "kmeans")
                    + 1
                )
            )
        )

        # Initialize the model
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
                trainer_params=self.trainer_params_user,
            ).fit(data)
            self.model_.means.copy_(estimator.model_.centroids)

        # Fit model
        module = GaussianMixtureLightningModule(
            self.model_,
            init_strategy=self.init_strategy,  # type: ignore
            convergence_tolerance=self.convergence_tolerance,
            covariance_regularization=self.covariance_regularization,
            batch_training=batch_training,
        )
        self.trainer_.fit(module, self._init_data_loader(data, for_training=True))

        # Assign convergence properties
        self.num_iter_ = module.current_epoch + 1
        if batch_training:
            # For batch training, the actual number of iterations is lower. For random
            # initialization, this should be an even number, for kmeans initialization, it should
            # be an odd number.
            self.num_iter_ //= 2
        # For initialization, one optimization cycle is used. However, for batch training, this is
        # already absorbed for the kmeans initialization due to the integer division.
        if not (batch_training and self.init_strategy == "kmeans"):
            self.num_iter_ -= 1

        self.converged_ = self.trainer_.should_stop
        self.nll_ = cast(float, self.trainer_.callback_metrics["nll"].item())
        return self

    def sample(self, num_datapoints: int) -> torch.Tensor:
        """
        Samples datapoints from the fitted Gaussian mixture.

        Args:
            num_datapoints: The number of datapoints to sample.

        Returns:
            A tensor of shape ``[num_datapoints, dim]`` providing the samples.

        Note:
            This method does not parallelize across multiple processes, i.e. performs no
            synchronization.
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
        result = self.trainer_.test(
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

        Attention:
            When calling this function in a multi-process environment, each process receives only
            a subset of the predictions. If you want to aggregate predictions, make sure to gather
            the values returned from this method.
        """
        result = self.trainer_.predict(
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

        Attention:
            When calling this function in a multi-process environment, each process receives only
            a subset of the predictions. If you want to aggregate predictions, make sure to gather
            the values returned from this method.
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

        Attention:
            When calling this function in a multi-process environment, each process receives only
            a subset of the predictions. If you want to aggregate predictions, make sure to gather
            the values returned from this method.
        """
        result = self.trainer_.predict(
            GaussianMixtureLightningModule(self.model_),
            self._init_data_loader(data, for_training=False),
        )
        return torch.cat([x[0] for x in cast(List[Tuple[torch.Tensor, torch.Tensor]], result)])