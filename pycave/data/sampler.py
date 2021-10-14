import math
from typing import Iterator
import torch
from torch.utils.data import Sampler


class TensorBatchSampler(Sampler[slice]):
    """
    Sampler providing batches of contiguous indices. These indices can be used for constructing
    batches swiftly.

    Note:
        This sampler should only be used within a single process. Currently, it does not support
        shuffling.
    """

    def __init__(self, data_source: torch.Tensor, batch_size: int):
        """
        Args:
            data_source: A tensor of shape ``[num_datapoints, *]`` providing the items. Used to
                derive the number of items.
            batch_size: The number of items to sample for each batch.
        """
        super().__init__(data_source)
        self.dataset_size = data_source.size(0)
        self.batch_size = batch_size

    def __len__(self) -> int:
        return math.ceil(self.dataset_size / self.batch_size)

    def __iter__(self) -> Iterator[slice]:
        for i in range(len(self)):
            yield slice(i * self.batch_size, (i + 1) * self.batch_size)


class DistributedTensorBatchSampler(Sampler[torch.Tensor]):
    """
    Sampler providing batches of sampled indices in a distributed environment. If the data is not
    divisible by the number of processes, this sampler yields randomly selected duplicate items.
    This way, it can be ensured that every process runs equally many batches. This sampler
    therefore always shuffles the data.
    """

    def __init__(
        self,
        data_source: torch.Tensor,
        batch_size: int,
        num_replicas: int,
        rank: int,
        seed: int = 0,
    ):
        """
        Args:
            data_source: A tensor of shape ``[num_datapoints, *]`` providing the items. Used to
                derive the number of items.
            batch_size: The number of items to sample for each batch.
            num_replicas: The total number of processes for which the sampler is used (i.e. the
                world size).
            rank: The rank of the process for which this sampler is providing items.
            seed: The seed to use for sampling indices.
        """
        super().__init__(data_source)

        self.dataset_size = data_source.size(0)

        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank

        self.seed = seed
        self.current_epoch = 0

    def __len__(self) -> int:
        return math.ceil(math.ceil(self.dataset_size / self.num_replicas) / self.batch_size)

    def __iter__(self) -> Iterator[torch.Tensor]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.current_epoch)

        permutation = torch.randperm(self.dataset_size, generator=g)
        chunk_size = math.ceil(self.dataset_size / self.num_replicas)
        local_choices = permutation[chunk_size * self.rank : chunk_size * (self.rank + 1)]
        if len(local_choices) != chunk_size:
            # This happens if the data is not divisible by the number of replicas. We need to
            # choose a random element in each process. We just do this by picking the first
            # elements of the permutation.
            num_unfulfilled = chunk_size * self.num_replicas - self.dataset_size
            num_fulfilled = self.num_replicas - num_unfulfilled
            idx = self.rank - num_fulfilled
            local_choices = torch.cat([local_choices, permutation[idx : idx + 1]])

        for i in range(len(self)):
            yield local_choices[i * self.batch_size : (i + 1) * self.batch_size]

    def set_epoch(self, epoch: int) -> None:
        """
        Sets the epoch for this sampler to ensure that different indices are sampled in each epoch.
        """
        self.current_epoch = epoch


class UnrepeatedDistributedTensorBatchSampler(Sampler[slice]):
    """
    Sampler providing contiguous indices with possibly unevenly sized batches. This class is
    similar to :class:`DistributedTensorBatchSampler`. However, it yields items in-order and does
    not return duplicate items, possibly yielding unequally many batches for different parallel
    replicas. Thus, this should only be used for testing where it can be ensured that each replica
    receives at least one batch. See also
    :class:`pytorch_lightning.overrides.distributed.UnrepeatedDistributedSampler`.
    """

    def __init__(
        self,
        data_source: torch.Tensor,
        batch_size: int,
        num_replicas: int,
        rank: int,
    ):
        """
        Args:
            data_source: A tensor of shape ``[num_datapoints, *]`` providing the items. Used to
                derive the number of items.
            batch_size: The number of items to sample for each batch.
            num_replicas: The total number of processes for which the sampler is used (i.e. the
                world size).
            rank: The rank of the process for which this sampler is providing items.
        """
        super().__init__(data_source)

        self.dataset_size = data_source.size(0)

        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank

    def __len__(self) -> int:
        # This only applies to rank 0, but that is fine for displaying a progress bar
        return math.ceil(math.ceil(self.dataset_size / self.num_replicas) / self.batch_size)

    def __iter__(self) -> Iterator[slice]:
        chunk_sizes = [
            self.dataset_size // self.num_replicas + int(i < self.dataset_size % self.num_replicas)
            for i in range(self.num_replicas)
        ]
        prev_size = sum(chunk_sizes[: self.rank])
        local_size = chunk_sizes[self.rank]

        for i in range(math.ceil(local_size / self.batch_size)):
            yield slice(
                prev_size + i * self.batch_size,
                min(prev_size + (i + 1) * self.batch_size, prev_size + local_size),
            )
