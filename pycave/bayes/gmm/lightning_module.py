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

logger = logging.getLogger(__name__)

InitStrategy = Literal["random", "kmeans"]


# pylint: disable=abstract-method
class GaussianMixtureLightningModule(NonparametricLightningModule):
    """
    Lightning module for training and evaluating a Gaussian mixture model.
    """

    def __init__(
        self,
        model: GaussianMixtureModel,
        tol: float = 1e-3,
        reg_covar: float = 1e-6,
        batch_training: bool = False,
        init_strategy: InitStrategy = "kmeans",
    ):
        """
        Args:
            model: The Gaussian mixture model to use for training/evaluation.
        """
        super().__init__()

        self.model = model
        self.tol = tol
        self.batch_training = batch_training
        self.init_strategy = init_strategy
        self.mean_update_remainder = 0 if init_strategy == "random" else 1

        # For batch training, we store a model copy such that we can "replay" responsibilities
        if batch_training:
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
            reg=reg_covar,
            dist_sync_fn=self.all_gather,
        )

        # Initialize metrics
        self.metric_nll = AverageMeter(dist_sync_fn=self.all_gather)

    def configure_callbacks(self) -> List[pl.Callback]:
        early_stopping = EarlyStopping(
            "nll",
            min_delta=self.tol,
            patience=3 if self.batch_training else 1,
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
        if self.current_epoch == 0:
            # In the first, epoch, we are *always* running initialization. For random
            # initialization, we set random responsibilities and then run M- and E-step as usual.
            # For kmeans initialization, responsibilities are computed from the cluster
            # assignments. Updating the means could be skipped, but we keep it for convenience.
            # This should be changed in the future though.
            if self.init_strategy == "random":
                logging.info("Sampling responsibilities randomly.")
                responsibilities = torch.rand(
                    batch.size(0), self.model.config.num_components, device=self.device
                )
                responsibilities = responsibilities / responsibilities.sum(1, keepdim=True)
            else:  # init_strategy == "kmeans"
                logging.info("Computing responsibilities from k-means assignments.")
                distances = torch.cdist(batch, self.model.means)
                assignments = distances.argmin(1)
                onehot = torch.eye(self.model.config.num_components, device=batch.device)
                responsibilities = onehot[assignments]

            logging.info("Starting optimization.")
        else:
            # In any other case, we run the true E-step
            if not self.batch_training or self.current_epoch % 2 == self.mean_update_remainder:
                log_responsibilities, log_probs = self.model.forward(batch)
            else:
                log_responsibilities, log_probs = self.model_copy.forward(batch)
            responsibilities = log_responsibilities.exp()

            # Here, we can also log the NLL for early stopping
            nll = -log_probs.mean()
            self.metric_nll.update(nll)
            self.log("nll", self.metric_nll, on_step=False, on_epoch=True, prog_bar=True)

        ### (Partial) M-Step
        # If we don't do batching, it's easy. We can just update all parameters.
        # If we do batching, however, we need to be careful:
        # - For random initialization, we run a full "fuzzy" update where we use intermediate
        #   means to compute an estimate of the covariance matrix.
        # - For kmeans initialization, we update the covariances.
        # - Otherwise, we update priors and means for even epochs and covariances for odd epochs.
        if not self.batch_training or (self.current_epoch == 0 and self.init_strategy == "random"):
            self.prior_aggregator.update(responsibilities)
            means = self.mean_aggregator.forward(batch, responsibilities)
            self.covar_aggregator.update(batch, responsibilities, means)
        elif self.current_epoch % 2 == self.mean_update_remainder:
            self.prior_aggregator.update(responsibilities)
            self.mean_aggregator.update(batch, responsibilities)
        else:
            self.covar_aggregator.update(batch, responsibilities, self.model.means)

    def nonparametric_training_epoch_end(self) -> None:
        # Prior to updating the model, we might need to copy it in the case of batch training
        if self.batch_training and self.current_epoch % 2 == self.mean_update_remainder:
            self.model_copy.load_state_dict(self.model.state_dict())

        # Finalize the M-Step. When not training for batches or the current epoch is even (and we
        # don't do kmeans initialization), we can update the priors and means. Otherwise, we do the
        # covariances.
        if not self.batch_training or (self.current_epoch % 2 == self.mean_update_remainder):
            self.model.component_probs.copy_(self.prior_aggregator.compute())
            self.model.means.copy_(self.mean_aggregator.compute())
            # print(self.model.means)

        if (
            not self.batch_training
            or (self.current_epoch == 0 and self.init_strategy == "random")
            or self.current_epoch % 2 != self.mean_update_remainder
        ):
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
