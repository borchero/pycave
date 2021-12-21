from __future__ import annotations
import logging
from typing import Any, cast, Dict, List, Optional, Tuple
import torch
from lightkit import BaseEstimator
from lightkit.data import collate_tensor, DataLoader, dataset_from_tensors, TensorLike
from lightkit.estimator import PredictorMixin
from pycave.bayes.core import CovarianceType
from pycave.clustering import KMeans
from .lightning_module import (
    GaussianMixtureKmeansInitLightningModule,
    GaussianMixtureLightningModule,
    GaussianMixtureRandomInitLightningModule,
)
from .model import GaussianMixtureModel, GaussianMixtureModelConfig
from .types import GaussianMixtureInitStrategy

logger = logging.getLogger(__name__)


class GaussianMixture(
    BaseEstimator[GaussianMixtureModel],
    PredictorMixin[TensorLike, torch.Tensor],
):
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
        init_means: Optional[torch.Tensor] = None,
        convergence_tolerance: float = 1e-3,
        covariance_regularization: float = 1e-6,
        batch_size: Optional[int] = None,
        trainer_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            num_components: The number of components in the GMM. The dimensionality of each
                component is automatically inferred from the data.
            covariance_type: The type of covariance to assume for all Gaussian components.
            init_strategy: The strategy for initializing component means and covariances.
            init_means: An optional initial guess for the means of the components. If provided,
                must be a tensor of shape ``[num_components, num_features]``. If this is given,
                the ``init_strategy`` is ignored and the means are handled as if K-means
                initialization has been run.
            convergence_tolerance: The change in the per-datapoint negative log-likelihood which
                implies that training has converged.
            covariance_regularization: A small value which is added to the diagonal of the
                covariance matrix to ensure that it is positive semi-definite.
            batch_size: The batch size to use when fitting the model. If not provided, the full
                data will be used as a single batch. Set this if the full data does not fit into
                memory.
            num_workers: The number of workers to use for loading the data. Only used if a PyTorch
                dataset is passed to :meth:`fit` or related methods.
            trainer_params: Initialization parameters to use when initializing a PyTorch Lightning
                trainer. By default, it disables various stdout logs unless PyCave is configured to
                do verbose logging. Checkpointing and logging are disabled regardless of the log
                level. This estimator further sets the following overridable defaults:

                - ``max_epochs=100``

        Note:
            The number of epochs passed to the initializer only define the number of optimization
            epochs. Prior to that, initialization is run which may perform additional iterations
            through the data.

        Note:
            For batch training, the number of epochs run (i.e. the number of passes through the
            data), does not align with the number of epochs passed to the initializer. This is
            because the EM algorithm needs to be split up across two epochs. The actual number of
            minimum/maximum epochs is, thus, doubled. Nonetheless, :attr:`num_iter_` indicates how
            many EM iterations have been run.
        """
        super().__init__(
            default_params=dict(max_epochs=100),
            user_params=trainer_params,
        )

        self.num_components = num_components
        self.covariance_type = covariance_type
        self.init_strategy = init_strategy
        self.init_means = init_means
        self.convergence_tolerance = convergence_tolerance
        self.covariance_regularization = covariance_regularization

        self.batch_size = batch_size

    def fit(self, data: TensorLike) -> GaussianMixture:
        """
        Fits the Gaussian mixture on the provided data, estimating component priors, means and
        covariances. Parameters are estimated using the EM algorithm.

        Args:
            data: The tabular data to fit on. The dimensionality of the Gaussian mixture is
                automatically inferred from this data.

        Returns:
            The fitted Gaussian mixture.
        """
        # Initialize the model
        num_features = len(data[0])
        config = GaussianMixtureModelConfig(
            num_components=self.num_components,
            num_features=num_features,
            covariance_type=self.covariance_type,  # type: ignore
        )
        self._model = GaussianMixtureModel(config)

        # Setup the data loading
        loader = DataLoader(
            dataset_from_tensors(data),
            batch_size=self.batch_size or len(data),
            collate_fn=collate_tensor,
        )
        is_batch_training = self._num_batches_per_epoch(loader) == 1

        # Run k-means if required or copy means
        if self.init_means is not None:
            self.model_.means.copy_(self.init_means)
        elif self.init_strategy in ("kmeans", "kmeans++"):
            logger.info("Fitting K-means estimator...")
            params = self.trainer_params_user
            if self.init_strategy == "kmeans++":
                params = {**(params or {}), **dict(max_epochs=0)}

            estimator = KMeans(
                self.num_components,
                batch_size=self.batch_size,
                trainer_params=params,
            ).fit(data)
            self.model_.means.copy_(estimator.model_.centroids)

        # Run initialization
        logger.info("Running initialization...")
        if self.init_strategy in ("kmeans", "kmeans++") and self.init_means is None:
            module = GaussianMixtureKmeansInitLightningModule(
                self.model_,
                covariance_regularization=self.covariance_regularization,
            )
            self.trainer(max_epochs=1).fit(module, loader)
        else:
            module = GaussianMixtureRandomInitLightningModule(
                self.model_,
                covariance_regularization=self.covariance_regularization,
                is_batch_training=is_batch_training,
                use_model_means=self.init_means is not None,
            )
            self.trainer(max_epochs=1 + int(is_batch_training)).fit(module, loader)

        # Fit model
        logger.info("Fitting Gaussian mixture...")
        module = GaussianMixtureLightningModule(
            self.model_,
            convergence_tolerance=self.convergence_tolerance,
            covariance_regularization=self.covariance_regularization,
            is_batch_training=is_batch_training,
        )
        trainer = self.trainer(
            max_epochs=cast(int, self.trainer_params["max_epochs"]) * (1 + int(is_batch_training))
        )
        trainer.fit(module, loader)

        # Assign convergence properties
        self.num_iter_ = module.current_epoch + 1
        if is_batch_training:
            self.num_iter_ //= 2
        self.converged_ = trainer.should_stop
        self.nll_ = cast(float, trainer.callback_metrics["nll"].item())
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

    def score(self, data: TensorLike) -> float:
        """
        Computes the average negative log-likelihood (NLL) of the provided datapoints.

        Args:
            data: The datapoints for which to evaluate the NLL.

        Returns:
            The average NLL of all datapoints.

        Note:
            See :meth:`score_samples` to obtain NLL values for individual datapoints.
        """
        loader = DataLoader(
            dataset_from_tensors(data),
            batch_size=self.batch_size or len(data),
            collate_fn=collate_tensor,
        )
        result = self.trainer().test(
            GaussianMixtureLightningModule(self.model_), loader, verbose=False
        )
        return result[0]["nll"]

    def score_samples(self, data: TensorLike) -> torch.Tensor:
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
        loader = DataLoader(
            dataset_from_tensors(data),
            batch_size=self.batch_size or len(data),
            collate_fn=collate_tensor,
        )
        result = self.trainer().predict(GaussianMixtureLightningModule(self.model_), loader)
        return torch.stack([x[1] for x in cast(List[Tuple[torch.Tensor, torch.Tensor]], result)])

    def predict(self, data: TensorLike) -> torch.Tensor:
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

    def predict_proba(self, data: TensorLike) -> torch.Tensor:
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
        loader = DataLoader(
            dataset_from_tensors(data),
            batch_size=self.batch_size or len(data),
            collate_fn=collate_tensor,
        )
        result = self.trainer().predict(GaussianMixtureLightningModule(self.model_), loader)
        return torch.cat([x[0] for x in cast(List[Tuple[torch.Tensor, torch.Tensor]], result)])
