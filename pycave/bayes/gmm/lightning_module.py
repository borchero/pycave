from __future__ import annotations
import logging
from typing import List, Literal, Tuple
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping
from torchmetrics import AverageMeter
from pycave.bayes.core import cholesky_precision
from pycave.core import NonparametricLightningModule
from .metrics import CovarianceAggregator, MeanAggregator, PriorAggregator
from .model import GaussianMixtureModel
from .types import GaussianMixtureInitStrategy

logger = logging.getLogger(__name__)


# pylint: disable=abstract-method
class GaussianMixtureLightningModule(NonparametricLightningModule):
    """
    Lightning module for training and evaluating a Gaussian mixture model.
    """

    def __init__(
        self,
        model: GaussianMixtureModel,
        init_strategy: GaussianMixtureInitStrategy = "kmeans",
        convergence_tolerance: float = 1e-3,
        covariance_regularization: float = 1e-6,
        batch_training: bool = False,
    ):
        """
        Args:
            model: The Gaussian mixture model to use for training/evaluation.
            init_strategy: The strategy for initializing model parameters.
            convergence_tolerance: The change in the per-datapoint negative log-likelihood which
                implies that training has converged.
            covariance_regularization: A small value which is added to the diagonal of the
                covariance matrix to ensure that it is positive semi-definite.
            batch_training: Whether training is performed on mini-batches instead of the entire
                data at once. In the case of batching, the EM-algorithm is "split" across two
                epochs.
        """
        super().__init__()

        self.model = model
        self.convergence_tolerance = convergence_tolerance
        self.batch_training = batch_training
        self.init_strategy = init_strategy

        # For batch training, we store a model copy such that we can "replay" responsibilities
        if self.batch_training:
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
        early_stopping = EarlyStopping(
            "nll",
            min_delta=self.convergence_tolerance,
            patience=2 if self.batch_training else 1,
            check_on_train_epoch_end=True,
            strict=False,
        )
        return [early_stopping]

    def on_train_epoch_start(self) -> None:
        self.prior_aggregator.reset()
        self.mean_aggregator.reset()
        self.covar_aggregator.reset()

    def nonparametric_training_step(self, batch: torch.Tensor, _batch_idx: int) -> None:
        ### E-Step
        if self._is_in_initialization_phase:
            # In the first, epoch, we are *always* running initialization. For random
            # initialization, we set random responsibilities and then run M- and E-step as usual.
            # For kmeans initialization, responsibilities are computed from the cluster
            # assignments. Updating the means could be skipped, but we keep it for convenience.
            # This should be changed in the future though.
            if self.init_strategy == "random" and self.current_epoch == 0:
                logging.info("Sampling responsibilities randomly.")
                responsibilities = torch.rand(
                    batch.size(0), self.model.config.num_components, device=self.device
                )
                responsibilities = responsibilities / responsibilities.sum(1, keepdim=True)
            else:  # init_strategy == "kmeans" or self.current_epoch > 0
                # We assign clusters deterministically even for random initialization to get
                # useful values for the covariance
                logging.info("Computing responsibilities from one-hot assignments.")
                distances = torch.cdist(batch, self.model.means)
                assignments = distances.argmin(1)
                onehot = torch.eye(self.model.config.num_components, device=batch.device)
                responsibilities = onehot[assignments]

            if self.current_epoch == 1 or self.init_strategy == "kmeans":
                logging.info("Starting optimization.")
        else:
            # In any other case, we run the true E-step
            if self._computes_responsibilities_on_live_model:
                log_responsibilities, log_probs = self.model.forward(batch)
            else:
                log_responsibilities, log_probs = self.model_copy.forward(batch)
            responsibilities = log_responsibilities.exp()

            # Here, we can also log the NLL for early stopping. If we're running batch training, we
            # only do this in epochs where we update the covariance.
            if self._computes_nll:
                self.metric_nll.update(-log_probs)
                self.log("nll", self.metric_nll, on_step=False, on_epoch=True, prog_bar=True)

        ### (Partial) M-Step
        # If we don't do batching, it's easy. We can just update all parameters.
        # If we do batching, however, we need to be careful:
        # - For random initialization, we run a full "fuzzy" update where we use intermediate
        #   means to compute an estimate of the covariance matrix.
        # - For kmeans initialization, we update the covariances.
        # - Otherwise, we update priors and means for even epochs and covariances for odd epochs.
        if "priors" in self._update_parameters:
            self.prior_aggregator.update(responsibilities)
        if "means" in self._update_parameters:
            means = self.mean_aggregator.forward(batch, responsibilities)
            if "covars" in self._update_parameters:
                self.covar_aggregator.update(batch, responsibilities, means)
        elif "covars" in self._update_parameters:
            self.covar_aggregator.update(batch, responsibilities, self.model.means)

    def nonparametric_training_epoch_end(self) -> None:
        # Prior to updating the model, we might need to copy it in the case of batch training
        if self._requires_to_copy_live_model:
            self.model_copy.load_state_dict(self.model.state_dict())

        # Finalize the M-Step. When not training for batches or the current epoch is even (and we
        # don't do kmeans initialization), we can update the priors and means. Otherwise, we do the
        # covariances.
        if "priors" in self._update_parameters:
            self.model.component_probs.copy_(self.prior_aggregator.compute())

        if "means" in self._update_parameters:
            self.model.means.copy_(self.mean_aggregator.compute())

        if "covars" in self._update_parameters:
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

    # ---------------------------------------------------------------------------------------------
    @property
    def _is_in_initialization_phase(self) -> bool:
        if self.batch_training and self.init_strategy == "random":
            return self.current_epoch < 2
        return self.current_epoch == 0

    @property
    def _computes_responsibilities_on_live_model(self) -> bool:
        assert self.current_epoch != 0
        if not self.batch_training:
            return True
        return self.current_epoch % 2 == (0 if self.init_strategy == "random" else 1)

    @property
    def _requires_to_copy_live_model(self) -> bool:
        if not self.batch_training:
            return False
        return self.current_epoch % 2 == (0 if self.init_strategy == "random" else 1)

    @property
    def _computes_nll(self) -> bool:
        if self.current_epoch == 0:
            return False
        if not self.batch_training:
            return True
        return self.current_epoch % 2 == (1 if self.init_strategy == "random" else 0)

    @property
    def _update_parameters(self) -> List[Literal["priors", "means", "covars"]]:
        if not self.batch_training:
            return ["priors", "means", "covars"]
        if self.current_epoch == 0 and self.init_strategy == "kmeans":
            return ["priors", "covars"]
        if self.current_epoch % 2 == (0 if self.init_strategy == "random" else 1):
            return ["priors", "means"]
        return ["covars"]
