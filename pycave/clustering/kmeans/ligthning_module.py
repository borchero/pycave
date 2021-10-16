from typing import List, Optional
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping
from torchmetrics import AverageMeter
from pycave.core import NonparametricLightningModule
from .metrics import CentroidAggregator, DistanceSampler, UniformSampler
from .model import KMeansModel
from .types import KMeansInitStrategy


# pylint: disable=abstract-method
class KMeansLightningModule(NonparametricLightningModule):
    """
    The lightning module...
    """

    def __init__(
        self,
        model: KMeansModel,
        init_strategy: KMeansInitStrategy = "kmeans++",
        tol: float = 1e-4,
        batch_training: bool = False,
    ):
        """
        Args:
            model:
        """
        super().__init__()

        self.model = model
        self.init_strategy = init_strategy
        self.tol = tol
        self.batch_training = batch_training

        # Initialize aggregators
        self.init_uniform_sampler = UniformSampler(
            num_choices=self.model.config.num_clusters if self.init_strategy == "random" else 1,
            num_features=self.model.config.num_features,
            dist_sync_fn=self.all_gather,
        )
        if self.init_strategy == "kmeans++":
            self.init_distance_sampler = DistanceSampler(
                num_features=self.model.config.num_features,
                dist_sync_fn=self.all_gather,
            )

        self.centroid_aggregator = CentroidAggregator(
            num_clusters=self.model.config.num_clusters,
            num_features=self.model.config.num_features,
            dist_sync_fn=self.all_gather,
        )

        # Initialize metrics
        self.metric_inertia = AverageMeter()

        # Initialize buffers
        if not self.batch_training and self.init_strategy == "kmeans++":
            # If we're not training on batches, we assume that distances can be cached in memory.
            # The buffer will be assigned the correct size in the first step.
            self.distance_cache: torch.Tensor
            self.register_buffer("distance_cache", torch.empty(1), persistent=False)

    def configure_callbacks(self) -> List[pl.Callback]:
        early_stopping = EarlyStopping(
            "inertia",
            min_delta=self.tol,
            patience=1,
            check_on_train_epoch_end=True,
            strict=False,  # disables early stopping as long as inertia is not set
        )
        return [early_stopping]

    def on_train_epoch_start(self) -> None:
        if self._is_in_initialization_phase:
            if self.current_epoch == 0:
                self.init_uniform_sampler.reset()
            else:
                self.init_distance_sampler.reset()
        else:
            self.centroid_aggregator.reset()

    def nonparametric_training_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        if self._is_in_initialization_phase:
            if self.current_epoch == 0:
                # In the first epoch, we do uniform sampling, no matter the initialization strategy
                self.init_uniform_sampler.update(batch)
                if not self.batch_training:
                    self.distance_cache = batch.new_empty(
                        batch.size(0), self.model.config.num_clusters
                    )
            else:
                # Otherwise, we're doing kmeans++ initialization. If we're doing batch training, we
                # compute all distances since we can't cache (complexity of kmeans++ quadratic in
                # the number of clusters). Otherwise, we can just compute the next one (complexity
                # of kmeans++ linear in the number of clusters).
                if self.batch_training:
                    distances = torch.cdist(batch, self.model.centroids[: self.current_epoch])
                else:
                    idx = slice(self.current_epoch - 1, self.current_epoch)
                    self.distance_cache[:, idx] = torch.cdist(batch, self.model.centroids[idx])
                    distances = self.distance_cache[:, : self.current_epoch]
                self.init_distance_sampler.update(batch, distances)
        else:
            # When we're not in initialization phase, we first compute the cluster assignments
            distances, inertias = self.model.forward(batch)
            assignments = distances.argmin(1)

            # Then, we update the centroids
            self.centroid_aggregator.update(batch, assignments)

            # And log the inertia
            self.metric_inertia.update(inertias)
            self.log("inertia", self.metric_inertia, on_step=False, on_epoch=True, prog_bar=True)

    def nonparametric_training_epoch_end(self) -> None:
        if self._is_in_initialization_phase:
            if self.init_strategy == "random":
                centroids = self.init_uniform_sampler.compute()
                centroids = self._gather_and_choose_first(centroids)
                self.model.centroids.copy_(centroids)
            else:
                if self.current_epoch == 0:
                    centroid = self.init_uniform_sampler.compute()[0]
                else:
                    centroid = self.init_distance_sampler.compute()
                centroid = self._gather_and_choose_first(centroid)
                self.model.centroids[self.current_epoch].copy_(centroid)
        else:
            centroids = self.centroid_aggregator.compute()
            self.model.centroids.copy_(centroids)

    def test_step(self, batch: torch.Tensor, _batch_idx: int) -> None:
        _, inertias = self.model.forward(batch)
        self.metric_inertia.update(inertias)
        self.log("inertia", self.metric_inertia)

    def predict_step(
        self, batch: torch.Tensor, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> torch.Tensor:
        distances, _ = self.model.forward(batch)
        return distances

    # ---------------------------------------------------------------------------------------------

    def _gather_and_choose_first(self, x: torch.Tensor) -> torch.Tensor:
        gathered = self.all_gather(x)
        if gathered.dim() > x.dim():
            return gathered[0]
        return x

    @property
    def _is_in_initialization_phase(self) -> bool:
        if self.init_strategy == "random":
            return self.current_epoch == 0
        # kmeans++ initialization - first k epochs are used to iteratively select seeds
        return self.current_epoch < self.model.config.num_clusters
