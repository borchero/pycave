import random
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
        if self.num_choices == 1:
            # If there is only one choice, the fastest thing is to use the `random` package. The
            # cumulative weight of the data is its size, the cumulative weight of the current
            # choice is some value.
            cum_weight = data.size(0) + self.choice_weights.item()
            if random.random() * cum_weight < data.size(0):
                # Use some item from the data, else keep the current choice
                self.choices.copy_(data[random.randrange(data.size(0))])
        else:
            # The choices are computed from scratch every time, weighting the current choices by
            # the cumulative weight put on them
            weights = torch.cat(
                [
                    torch.ones(data.size(0), device=data.device, dtype=data.dtype),
                    self.choice_weights,
                ]
            )
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
    choices is not part of this sampler. Within each "cycle", it computes a given number of
    candidates. Candidates are sampled independently and may be duplicates.
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
        self.num_features = num_features

        self.choices: torch.Tensor
        self.add_state("choices", torch.empty(num_choices, num_features), dist_reduce_fx="cat")

        # Cumulative distance is the same for all choices
        self.cumulative_squared_distance: torch.Tensor
        self.add_state("cumulative_squared_distance", torch.zeros(1), dist_reduce_fx="cat")

    def update(self, data: torch.Tensor, shortest_distances: torch.Tensor) -> None:
        eps = torch.finfo(data.dtype).eps
        squared_distances = shortest_distances.square()

        # For all choices, check if we should use a sample from the data or the existing choice
        data_dist = squared_distances.sum()
        cum_dist = data_dist + eps + self.cumulative_squared_distance
        use_choice_from_data = (
            torch.rand(self.num_choices, device=data.device, dtype=data.dtype) * cum_dist
            < data_dist + eps
        )

        # Then, we sample from the data `num_choices` times and replace if needed
        choices = (squared_distances + eps).multinomial(self.num_choices, replacement=True)
        self.choices.masked_scatter_(
            use_choice_from_data.unsqueeze(1), data[choices[use_choice_from_data]]
        )

        # In any case, the cumulative distances are updated
        self.cumulative_squared_distance.add_(data_dist)

    def compute(self) -> torch.Tensor:
        # Upon computation, we sample if there is more than one choice (ddp setting)
        if self.choices.size(0) > self.num_choices:
            # choices now have shape [num_choices, num_processes, num_features]
            choices = self.choices.reshape(-1, self.num_choices, self.num_features).transpose(0, 1)
            # For each choice, we sample across processes
            choice_indices = torch.arange(self.num_choices, device=self.choices.device)
            process_indices = self.cumulative_squared_distance.multinomial(
                self.num_choices, replacement=True
            )
            return choices[choice_indices, process_indices]
        # Otherwise, we can return the choices
        return self.choices


class BatchSummer(Metric):
    """
    Sums the values for a batch of items independently.
    """

    def __init__(self, num_values: int, *, dist_sync_fn: Optional[Callable[[Any], Any]] = None):
        super().__init__(dist_sync_fn=dist_sync_fn)  # type: ignore

        self.sums: torch.Tensor
        self.add_state("sums", torch.zeros(num_values), dist_reduce_fx="sum")

    def update(self, values: torch.Tensor) -> None:
        self.sums.add_(values.sum(0))

    def compute(self) -> torch.Tensor:
        return self.sums


class BatchAverager(Metric):
    """
    Averages the values for a batch of items independently.
    """

    def __init__(
        self,
        num_values: int,
        for_variance: bool,
        *,
        dist_sync_fn: Optional[Callable[[Any], Any]] = None,
    ):
        super().__init__(dist_sync_fn=dist_sync_fn)  # type: ignore

        self.for_variance = for_variance

        self.sums: torch.Tensor
        self.add_state("sums", torch.zeros(num_values), dist_reduce_fx="sum")

        self.counts: torch.Tensor
        self.add_state("counts", torch.zeros(num_values), dist_reduce_fx="sum")

    def update(self, values: torch.Tensor) -> None:
        self.sums.add_(values.sum(0))
        self.counts.add_(values.size(0))

    def compute(self) -> torch.Tensor:
        return self.sums / (self.counts - 1 if self.for_variance else self.counts)
