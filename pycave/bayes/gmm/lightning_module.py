from __future__ import annotations
from typing import List, Tuple
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping
from torchmetrics import AverageMeter
from pycave.bayes.core import cholesky_precision
from pycave.utils import NonparametricLightningModule
from .metrics import CovarianceAggregator, MeanAggregator, PriorAggregator
from .model import GaussianMixtureModel

# -------------------------------------------------------------------------------------------------
# TRAINING


class GaussianMixtureLightningModule(NonparametricLightningModule):
    """
    Lightning module for training and evaluating a Gaussian mixture model.
    """

    def __init__(
        self,
        model: GaussianMixtureModel,
        convergence_tolerance: float = 1e-3,
        covariance_regularization: float = 1e-6,
        is_batch_training: bool = False,
    ):
        """
        Args:
            model: The Gaussian mixture model to use for training/evaluation.
            convergence_tolerance: The change in the per-datapoint negative log-likelihood which
                implies that training has converged.
            covariance_regularization: A small value which is added to the diagonal of the
                covariance matrix to ensure that it is positive semi-definite.
            is_batch_training: Whether training is performed on mini-batches instead of the entire
                data at once. In the case of batching, the EM-algorithm is "split" across two
                epochs.
        """
        super().__init__()

        self.model = model
        self.convergence_tolerance = convergence_tolerance
        self.is_batch_training = is_batch_training

        # For batch training, we store a model copy such that we can "replay" responsibilities
        if self.is_batch_training:
            self.model_copy = GaussianMixtureModel(self.model.config)
            self.model_copy.load_state_dict(self.model.state_dict())

        # Initialize aggregators
        self.prior_aggregator = PriorAggregator(
            num_components=self.model.config.num_components,
            dist_sync_fn=self.all_gather,
        )
        self.mean_aggregator = MeanAggregator(
            num_components=self.model.config.num_components,
            num_features=self.model.config.num_features,
            dist_sync_fn=self.all_gather,
        )
        self.covar_aggregator = CovarianceAggregator(
            num_components=self.model.config.num_components,
            num_features=self.model.config.num_features,
            covariance_type=self.model.config.covariance_type,
            reg=covariance_regularization,
            dist_sync_fn=self.all_gather,
        )

        # Initialize metrics
        self.metric_nll = AverageMeter(dist_sync_fn=self.all_gather)

    def configure_callbacks(self) -> List[pl.Callback]:
        if self.convergence_tolerance == 0:
            return []
        early_stopping = EarlyStopping(
            "nll",
            min_delta=self.convergence_tolerance,
            patience=2 if self.is_batch_training else 1,
            check_on_train_epoch_end=True,
            strict=False,  # Allows to not log every epoch
        )
        return [early_stopping]

    def on_train_epoch_start(self) -> None:
        self.prior_aggregator.reset()
        self.mean_aggregator.reset()
        self.covar_aggregator.reset()

    def nonparametric_training_step(self, batch: torch.Tensor, _batch_idx: int) -> None:
        ### E-Step
        if self._computes_responsibilities_on_live_model:
            log_responsibilities, log_probs = self.model.forward(batch)
        else:
            log_responsibilities, log_probs = self.model_copy.forward(batch)
        responsibilities = log_responsibilities.exp()

        # Compute the NLL for early stopping
        if self._should_log_nll:
            self.metric_nll.update(-log_probs)
            self.log("nll", self.metric_nll, on_step=False, on_epoch=True, prog_bar=True)

        ### (Partial) M-Step
        if self._should_update_means:
            self.prior_aggregator.update(responsibilities)
            self.mean_aggregator.update(batch, responsibilities)
            if self._should_update_covars:
                means = self.mean_aggregator.compute()
                self.covar_aggregator.update(batch, responsibilities, means)
        else:
            self.covar_aggregator.update(batch, responsibilities, self.model.means)

    def nonparametric_training_epoch_end(self) -> None:
        # Prior to updating the model, we might need to copy it in the case of batch training
        if self._requires_to_copy_live_model:
            self.model_copy.load_state_dict(self.model.state_dict())

        # Finalize the M-Step
        if self._should_update_means:
            priors = self.prior_aggregator.compute()
            self.model.component_probs.copy_(priors)

            means = self.mean_aggregator.compute()
            self.model.means.copy_(means)

        if self._should_update_covars:
            covars = self.covar_aggregator.compute()
            self.model.precisions_cholesky.copy_(
                cholesky_precision(covars, self.model.config.covariance_type)
            )

    def test_step(self, batch: torch.Tensor, _batch_idx: int) -> None:
        _, log_probs = self.model.forward(batch)
        self.metric_nll.update(-log_probs)
        self.log("nll", self.metric_nll)

    def predict_step(
        self, batch: torch.Tensor, batch_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        log_responsibilities, log_probs = self.model.forward(batch)
        return log_responsibilities.exp(), -log_probs

    @property
    def _computes_responsibilities_on_live_model(self) -> bool:
        if not self.is_batch_training:
            return True
        return self.current_epoch % 2 == 0

    @property
    def _requires_to_copy_live_model(self) -> bool:
        if not self.is_batch_training:
            return False
        return self.current_epoch % 2 == 0

    @property
    def _should_log_nll(self) -> bool:
        if not self.is_batch_training:
            return True
        return self.current_epoch % 2 == 1

    @property
    def _should_update_means(self) -> bool:
        if not self.is_batch_training:
            return True
        return self.current_epoch % 2 == 0

    @property
    def _should_update_covars(self) -> bool:
        if not self.is_batch_training:
            return True
        return self.current_epoch % 2 == 1


# -------------------------------------------------------------------------------------------------
# INIT STRATEGIES


class GaussianMixtureKmeansInitLightningModule(NonparametricLightningModule):
    """
    Lightning module for initializing a Gaussian mixture from centroids found via K-Means.
    """

    def __init__(self, model: GaussianMixtureModel, covariance_regularization: float):
        """
        Args:
            model: The model whose parameters to initialize.
            covariance_regularization: A small value which is added to the diagonal of the
                covariance matrix to ensure that it is positive semi-definite.
        """
        super().__init__()

        self.model = model

        self.prior_aggregator = PriorAggregator(
            num_components=self.model.config.num_components,
            dist_sync_fn=self.all_gather,
        )
        self.covar_aggregator = CovarianceAggregator(
            num_components=self.model.config.num_components,
            num_features=self.model.config.num_features,
            covariance_type=self.model.config.covariance_type,
            reg=covariance_regularization,
            dist_sync_fn=self.all_gather,
        )

    def on_train_epoch_start(self) -> None:
        self.prior_aggregator.reset()
        self.covar_aggregator.reset()

    def nonparametric_training_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        # Just like for k-means, responsibilities are one-hot assignments to the clusters
        responsibilities = _one_hot_responsibilities(batch, self.model.means)

        # Then, we can update the aggregators
        self.prior_aggregator.update(responsibilities)
        self.covar_aggregator.update(batch, responsibilities, self.model.means)

    def nonparametric_training_epoch_end(self) -> None:
        priors = self.prior_aggregator.compute()
        self.model.component_probs.copy_(priors)

        covars = self.covar_aggregator.compute()
        self.model.precisions_cholesky.copy_(
            cholesky_precision(covars, self.model.config.covariance_type)
        )


class GaussianMixtureRandomInitLightningModule(NonparametricLightningModule):
    """
    Lightning module for initializing a Gaussian mixture randomly or using the assignments for
    arbitrary means that were not found via K-means. For batch training, this requires two epochs,
    otherwise, it requires a single epoch.
    """

    def __init__(
        self,
        model: GaussianMixtureModel,
        covariance_regularization: float,
        is_batch_training: bool,
        use_model_means: bool,
    ):
        """
        Args:
            model: The model whose parameters to initialize.
            covariance_regularization: A small value which is added to the diagonal of the
                covariance matrix to ensure that it is positive semi-definite.
            is_batch_training: Whether training is performed on mini-batches instead of the entire
                data at once.
            use_model_means: Whether the model's means ought to be used for one-hot component
                assignments.
        """
        super().__init__()

        self.model = model
        self.is_batch_training = is_batch_training
        self.use_model_means = use_model_means

        self.prior_aggregator = PriorAggregator(
            num_components=self.model.config.num_components,
            dist_sync_fn=self.all_gather,
        )
        self.mean_aggregator = MeanAggregator(
            num_components=self.model.config.num_components,
            num_features=self.model.config.num_features,
            dist_sync_fn=self.all_gather,
        )
        self.covar_aggregator = CovarianceAggregator(
            num_components=self.model.config.num_components,
            num_features=self.model.config.num_features,
            covariance_type=self.model.config.covariance_type,
            reg=covariance_regularization,
            dist_sync_fn=self.all_gather,
        )

        # For batch training, we store a model copy such that we can "replay" responsibilities
        if self.is_batch_training and self.use_model_means:
            self.model_copy = GaussianMixtureModel(self.model.config)
            self.model_copy.load_state_dict(self.model.state_dict())

    def on_train_epoch_start(self) -> None:
        self.prior_aggregator.reset()
        self.mean_aggregator.reset()
        self.covar_aggregator.reset()

    def nonparametric_training_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        if self.use_model_means:
            if self.current_epoch == 0:
                responsibilities = _one_hot_responsibilities(batch, self.model.means)
            else:
                responsibilities = _one_hot_responsibilities(batch, self.model_copy.means)
        else:
            responsibilities = torch.rand(
                batch.size(0),
                self.model.config.num_components,
                device=batch.device,
                dtype=batch.dtype,
            )
            responsibilities = responsibilities / responsibilities.sum(1, keepdim=True)

        if self.current_epoch == 0:
            self.prior_aggregator.update(responsibilities)
            self.mean_aggregator.update(batch, responsibilities)
            if not self.is_batch_training:
                means = self.mean_aggregator.compute()
                self.covar_aggregator.update(batch, responsibilities, means)
        else:
            # Only reached if batch training
            self.covar_aggregator.update(batch, responsibilities, self.model.means)

    def nonparametric_training_epoch_end(self) -> None:
        if self.current_epoch == 0 and self.is_batch_training:
            self.model_copy.load_state_dict(self.model.state_dict())

        if self.current_epoch == 0:
            priors = self.prior_aggregator.compute()
            self.model.component_probs.copy_(priors)

            means = self.mean_aggregator.compute()
            self.model.means.copy_(means)

        if (self.current_epoch == 0 and not self.is_batch_training) or self.current_epoch == 1:
            covars = self.covar_aggregator.compute()
            self.model.precisions_cholesky.copy_(
                cholesky_precision(covars, self.model.config.covariance_type)
            )


def _one_hot_responsibilities(data: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    distances = torch.cdist(data, centroids)
    assignments = distances.min(1).indices
    onehot = torch.eye(
        centroids.size(0),
        device=data.device,
        dtype=data.dtype,
    )
    return onehot[assignments]
