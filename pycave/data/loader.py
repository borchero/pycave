from typing import Callable, Generic, Iterator, Optional, TypeVar, Union
import torch
from torch.utils.data import BatchSampler, Sampler

T = TypeVar("T")


def _default_collate_fn(x: torch.Tensor) -> torch.Tensor:
    return x


class TensorDataLoader(Generic[T]):
    """
    Fast data loader for tabular data represented as tensors. This data loader drastically
    improves performance compared to PyTorch's built-in data loader by directly cutting batches
    from the data instead of getting individual items and stacking them.

    Note:
        This data loader does not provide options for multiprocessing since cutting from tensors is
        faster than sending tensors over queues. Also, it enforces the usage of a sampler instead
        of constructing one from init arguments.

    See also:
        - :class:`pycave.data.TensorBatchSampler`
        - :class:`pycave.data.DistributedTensorBatchSampler`
        - :class:`pycave.data.UnrepeatedDistributedTensorBatchSampler`
    """

    def __init__(
        self,
        dataset: torch.Tensor,
        sampler: Sampler[Union[slice, torch.Tensor]],
        collate_fn: Callable[[torch.Tensor], T] = _default_collate_fn,
    ):
        """
        Args:
            dataset: A tensor of shape ``[num_datapoints, *]`` representing all the available data.
            sampler: A batch sampler which provides either slices or batches of indices to gather
                from the dataset.
            collate_fn: A collation function which transforms a batch of items into another type.
                By default, it does not apply any transforms to the batch.
        """
        self.dataset = dataset
        self.sampler = sampler
        self.collate_fn = collate_fn

    @property
    def batch_sampler(self) -> Optional[BatchSampler]:
        """
        Returns a dummy ``None`` value to have parity with the ``DataLoader`` interface.
        """
        return None

    def __len__(self) -> int:
        return len(self.sampler)  # type: ignore

    def __iter__(self) -> Iterator[T]:
        for indices in self.sampler:
            yield self.collate_fn(self.dataset[indices])
