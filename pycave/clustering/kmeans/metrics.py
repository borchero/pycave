from typing import Any, Callable, Optional
import torch
from torchmetrics import Metric


class CentroidAggregator(Metric):
    """
    The centroid aggregator aggregates kmeans centroids over batches and processes.
    """

    def __init__(
        self,
        num_clusters: int,
        num_features: int,
        *,
        dist_sync_fn: Optional[Callable[[Any], Any]] = None,
    ):
        super().__init__(dist_sync_fn=dist_sync_fn)  # type: ignore

        self.num_clusters = num_clusters
        self.num_features = num_features

        self.centroids: torch.Tensor
        self.add_state("centroids", torch.zeros(num_clusters, num_features), dist_reduce_fx="sum")

        self.cluster_counts: torch.Tensor
        self.add_state("cluster_counts", torch.zeros(num_clusters), dist_reduce_fx="sum")

    def update(self, data: torch.Tensor, assignments: torch.Tensor) -> None:
        indices = assignments.unsqueeze(1).expand(-1, self.num_features)
        self.centroids.scatter_add_(0, indices, data)

        counts = assignments.bincount(minlength=self.num_clusters).float()
        self.cluster_counts.add_(counts)

    def compute(self) -> torch.Tensor:
        return self.centroids / self.cluster_counts.unsqueeze(-1)


class UniformSampler(Metric):
    """
    The uniform sampler randomly samples a specified number of datapoints uniformly from all
    datapoints. The idea is the following: sample the number of choices from each batch and
    track the number of datapoints that was already sampled from. When sampling from the union of
    existing choices and a new batch, more weight is put on the existing choices (according to the
    number of datapoints they were already sampled from).
    """

    def __init__(
        self,
        num_choices: int,
        num_features: int,
        *,
        dist_sync_fn: Optional[Callable[[Any], Any]] = None,
    ):
        super().__init__(dist_sync_fn=dist_sync_fn)  # type: ignore

        self.num_choices = num_choices

        self.choices: torch.Tensor
        self.add_state("choices", torch.empty(num_choices, num_features), dist_reduce_fx="cat")

        self.choice_weights: torch.Tensor
        self.add_state("choice_weights", torch.zeros(num_choices), dist_reduce_fx="cat")

    def update(self, data: torch.Tensor) -> None:
        # The choices are computed from scratch every time, weighting the current choices by the
        # cumulative weight put on them
        weights = torch.cat([torch.ones(data.size(0), device=data.device), self.choice_weights])
        pool = torch.cat([data, self.choices])
        samples = weights.multinomial(self.num_choices)
        self.choices.copy_(pool[samples])

        # The weights are the cumulative counts, divided by the number of choices
        self.choice_weights.add_(data.size(0) / self.num_choices)

    def compute(self) -> torch.Tensor:
        # In the ddp setting, there are "too many" choices, so we sample
        if self.choices.size(0) > self.num_choices:
            samples = self.choice_weights.multinomial(self.num_choices)
            return self.choices[samples]
        return self.choices


class DistanceSampler(Metric):
    """
    The distance sampler may be used for kmeans++ initialization, to iteratively select centroids
    according to their squared distances to existing choices. Computing the distance to existing
    choices is not part of this sampler. Within each "cycle", it computes a single choice.
    """

    def __init__(
        self,
        num_features: int,
        *,
        dist_sync_fn: Optional[Callable[[Any], Any]] = None,
    ):
        super().__init__(dist_sync_fn=dist_sync_fn)  # type: ignore

        self.choice: torch.Tensor
        self.add_state("choice", torch.empty(1, num_features), dist_reduce_fx="cat")

        self.cumulative_squared_distance: torch.Tensor
        self.add_state("cumulative_squared_distance", torch.zeros(1), dist_reduce_fx="cat")

    def update(self, data: torch.Tensor, distances: torch.Tensor) -> None:
        shortest_distances = distances.gather(1, distances.argmin(1, keepdim=True)).squeeze(1)
        squared_distances = shortest_distances.square()
        weights = torch.cat([self.cumulative_squared_distance, squared_distances])
        choice = weights.multinomial(1)
        if choice.item() != 0:
            # If the choice is not 0, one of the new datapoints is chosen
            self.choice.copy_(data[choice - 1])

        # In any case, the cumulative distances are updated
        self.cumulative_squared_distance.add_(squared_distances.sum())

    def compute(self) -> torch.Tensor:
        # Upon computation, we sample if there is more than one choice (ddp setting)
        if self.choice.size(0) > 1:
            choice = self.cumulative_squared_distance.multinomial(1)
            return self.choice[choice][0]
        # Otherwise, we can return the choice
        return self.choice[0]
