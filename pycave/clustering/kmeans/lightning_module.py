# pylint: disable=abstract-method
import math
from typing import List, Literal
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping
from torchmetrics import AverageMeter
from pycave.utils import NonparametricLightningModule
from .metrics import (
    BatchAverager,
    BatchSummer,
    CentroidAggregator,
    DistanceSampler,
    UniformSampler,
)
from .model import KMeansModel

# -------------------------------------------------------------------------------------------------
# TRAINING


class KMeansLightningModule(NonparametricLightningModule):
    """
    Lightning module for training and evaluating a K-Means model.
    """

    def __init__(
        self,
        model: KMeansModel,
        convergence_tolerance: float = 1e-4,
        predict_target: Literal["assignments", "distances", "inertias"] = "assignments",
    ):
        """
        Args:
            model: The model to train.
            convergence_tolerance: Training is conducted until the Frobenius norm of the change
                between cluster centroids falls below this threshold.
            predict_target: Whether to predict cluster assigments or distances to clusters.
        """
        super().__init__()

        self.model = model
        self.convergence_tolerance = convergence_tolerance
        self.predict_target = predict_target

        # Initialize aggregators
        self.centroid_aggregator = CentroidAggregator(
            num_clusters=self.model.config.num_clusters,
            num_features=self.model.config.num_features,
            dist_sync_fn=self.all_gather,
        )

        # Initialize metrics
        self.metric_inertia = AverageMeter()

    def configure_callbacks(self) -> List[pl.Callback]:
        if self.convergence_tolerance == 0:
            return []
        early_stopping = EarlyStopping(
            "frobenius_norm_change",
            patience=100000,
            stopping_threshold=self.convergence_tolerance,
            check_on_train_epoch_end=True,
        )
        return [early_stopping]

    def on_train_epoch_start(self) -> None:
        self.centroid_aggregator.reset()

    def nonparametric_training_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        # First, we compute the cluster assignments
        _, assignments, inertias = self.model.forward(batch)

        # Then, we update the centroids
        self.centroid_aggregator.update(batch, assignments)

        # And log the inertia
        self.metric_inertia.update(inertias)
        self.log("inertia", self.metric_inertia, on_step=False, on_epoch=True, prog_bar=True)

    def nonparametric_training_epoch_end(self) -> None:
        centroids = self.centroid_aggregator.compute()
        self.log("frobenius_norm_change", torch.linalg.norm(self.model.centroids - centroids))
        self.model.centroids.copy_(centroids)

    def test_step(self, batch: torch.Tensor, _batch_idx: int) -> None:
        _, _, inertias = self.model.forward(batch)
        self.metric_inertia.update(inertias)
        self.log("inertia", self.metric_inertia)

    def predict_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        distances, assignments, inertias = self.model.forward(batch)
        if self.predict_target == "assignments":
            return assignments
        if self.predict_target == "inertias":
            return inertias
        return distances


# -------------------------------------------------------------------------------------------------
# INIT STRATEGIES


class KmeansRandomInitLightningModule(NonparametricLightningModule):
    """
    Lightning module for initializing K-Means centroids randomly. Within the first epoch, all
    items are sampled. Thus, this module should only be trained for a single epoch.
    """

    def __init__(self, model: KMeansModel):
        """
        Args:
            model: The model to initialize.
        """
        super().__init__()

        self.model = model

        self.sampler = UniformSampler(
            num_choices=self.model.config.num_clusters,
            num_features=self.model.config.num_features,
            dist_sync_fn=self.all_gather_first,
        )

    def on_train_epoch_start(self) -> None:
        self.sampler.reset()

    def nonparametric_training_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        self.sampler.update(batch)

    def nonparametric_training_epoch_end(self) -> None:
        choices = self.sampler.compute()
        self.model.centroids.copy_(choices)


class KmeansPlusPlusInitLightningModule(NonparametricLightningModule):
    """
    Lightning module for K-Means++ initialization. It performs the following operations:

    - In the first epoch, a centroid is chosen at random.
    - In even epochs, candidates for the next centroid are sampled, based on the squared distance
      to their nearest cluster center.
    - In odd epochs, a candidate is selected deterministically as the next centroid.

    In total, initialization thus requires ``2 * k - 1`` epochs where ``k`` is the number of
    clusters.
    """

    def __init__(self, model: KMeansModel, is_batch_training: bool):
        """
        Args:
            model: The model to initialize.
            is_batch_training: Whether training is performed on mini-batches instead of the entire
                data at once.
        """
        super().__init__()

        self.model = model
        self.is_batch_training = is_batch_training

        self.uniform_sampler = UniformSampler(
            num_choices=1,
            num_features=self.model.config.num_features,
            dist_sync_fn=self.all_gather_first,
        )
        num_candidates = 2 + int(math.log(self.model.config.num_clusters))
        self.distance_sampler = DistanceSampler(
            num_choices=num_candidates,
            num_features=self.model.config.num_features,
            dist_sync_fn=self.all_gather_first,
        )
        self.candidate_inertia_summer = BatchSummer(
            num_candidates,
            dist_sync_fn=self.all_gather,
        )

        # Some buffers required for running initialization
        self.centroid_candidates: torch.Tensor
        self.register_buffer(
            "centroid_candidates",
            torch.empty(num_candidates, self.model.config.num_features),
            persistent=False,
        )

        if not self.is_batch_training:
            self.shortest_distance_cache: torch.Tensor
            self.register_buffer("shortest_distance_cache", torch.empty(1), persistent=False)

    def on_train_epoch_start(self) -> None:
        if self.current_epoch == 0:
            self.uniform_sampler.reset()
        elif self._is_current_epoch_sampling:
            self.distance_sampler.reset()
        else:
            self.candidate_inertia_summer.reset()

    def nonparametric_training_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        if self.current_epoch == 0:
            self.uniform_sampler.update(batch)
            return
        # In all other epochs, we either sample a number of candidates from the remaining
        # datapoints or select a candidate deterministically. In any case, the shortest
        # distance is  required.
        if self.current_epoch == 1:
            # In the first epoch, we can skip any argmin as the shortest distances are computed
            # with respect to the first centroid.
            shortest_distances = torch.cdist(batch, self.model.centroids[:1]).squeeze(1)
            if not self.is_batch_training:
                self.shortest_distance_cache = shortest_distances
        elif self.is_batch_training:
            # For batch training, we always need to recompute all distances since we can't
            # cache them (this is the whole reason for batch training).
            distances = torch.cdist(batch, self.model.centroids[: self._init_epoch + 1])
            shortest_distances = distances.gather(
                1, distances.min(1, keepdim=True).indices  # min is faster than argmin on CPU
            ).squeeze(1)
        else:
            # If we're not doing batch training, we only need to compute the distance to the
            # newest centroid (and only if we're currently sampling)
            if self._is_current_epoch_sampling:
                latest_distance = torch.cdist(
                    batch, self.model.centroids[self._init_epoch - 1].unsqueeze(0)
                ).squeeze(1)
                shortest_distances = torch.minimum(self.shortest_distance_cache, latest_distance)
                self.shortest_distance_cache = shortest_distances
            else:
                shortest_distances = self.shortest_distance_cache

        if self._is_current_epoch_sampling:
            # After computing the shortest distances, we can finally do the sampling
            self.distance_sampler.update(batch, shortest_distances)
        else:
            # Or, we select a candidate by the lowest resulting inertia
            distances = torch.cdist(batch, self.centroid_candidates)
            updated_distances = torch.minimum(distances, shortest_distances.unsqueeze(1))
            self.candidate_inertia_summer.update(updated_distances)

    def nonparametric_training_epoch_end(self) -> None:
        if self.current_epoch == 0:
            choice = self.uniform_sampler.compute()[0]
            self.model.centroids[0].copy_(choice)
        elif self._is_current_epoch_sampling:
            candidates = self.distance_sampler.compute()
            self.centroid_candidates.copy_(candidates)
        else:
            new_inertias = self.candidate_inertia_summer.compute()
            choice = new_inertias.argmin()
            self.model.centroids[self._init_epoch].copy_(self.centroid_candidates[choice])

    @property
    def _init_epoch(self) -> int:
        return (self.current_epoch + 1) // 2

    @property
    def _is_current_epoch_sampling(self) -> bool:
        return self.current_epoch % 2 == 1


# -------------------------------------------------------------------------------------------------
# MISC


class FeatureVarianceLightningModule(NonparametricLightningModule):
    """
    Lightning module for computing the average variance of a dataset's features. In the first
    epoch, it computes the features' means, then it can compute their variances.
    """

    def __init__(self, variances: torch.Tensor):
        """
        Args:
            variances: The output tensor where the variances are stored.
        """
        super().__init__()

        self.mean_aggregator = BatchAverager(
            num_values=variances.size(0),
            for_variance=False,
            dist_sync_fn=self.all_gather,
        )
        self.variance_aggregator = BatchAverager(
            num_values=variances.size(0),
            for_variance=True,
            dist_sync_fn=self.all_gather,
        )

        self.means: torch.Tensor
        self.register_buffer("means", torch.empty(variances.size(0)), persistent=False)

        self.variances: torch.Tensor
        self.register_buffer("variances", variances, persistent=False)

    def nonparametric_training_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        if self.current_epoch == 0:
            self.mean_aggregator.update(batch)
        else:
            self.variance_aggregator.update((batch - self.means.unsqueeze(0)).square())

    def nonparametric_training_epoch_end(self) -> None:
        if self.current_epoch == 0:
            self.means.copy_(self.mean_aggregator.compute())
        else:
            self.variances.copy_(self.variance_aggregator.compute())
