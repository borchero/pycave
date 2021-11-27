from typing import Callable, Generic, Iterator, Optional, Tuple, TypeVar, Union
import torch
from torch.utils.data import BatchSampler, Sampler
from .sampler import TensorBatchSampler

T = TypeVar("T")


def _default_collate_fn(
    x: Union[torch.Tensor, Tuple[torch.Tensor, ...]]
) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
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
        *tensors: torch.Tensor,
        batch_size: Optional[int] = None,
        sampler: Optional[Sampler[Union[slice, torch.Tensor]]] = None,
        collate_fn: Callable[
            [Union[torch.Tensor, Tuple[torch.Tensor, ...]]], T
        ] = _default_collate_fn,
    ):
        """
        Args:
            tensors: One or more tensors of shape ``[num_datapoints, *]``. For each index, this
                dataset returns all tensors' values at that index as tuples.
            batch_size: The batch size to use. Ignored if ``sampler`` is provided. If set to
                ``None``, each batch returns the full data.
            sampler: A batch sampler which provides either slices or batches of indices to gather
                from the dataset. By default, it uses :class:`pycave.data.TensorBatchSampler`.
            collate_fn: A collation function which transforms a batch of items into another type.
                By default, it does not apply any transforms to the batch.
        """
        assert len(tensors) > 0, "At least one tensor must be provided."

        self.tensors = tensors
        self.sampler = sampler or TensorBatchSampler(
            self.tensors[0], batch_size or self.tensors[0].size(0)
        )
        self.collate_fn = collate_fn

    @property
    def batch_sampler(self) -> Optional[BatchSampler]:
        """
        Returns a dummy ``None`` value for feature parity with the ``DataLoader`` interface.
        """
        return None

    def __len__(self) -> int:
        return len(self.sampler)  # type: ignore

    def __iter__(self) -> Iterator[T]:
        for indices in self.sampler:
            item = tuple(tensor[indices] for tensor in self.tensors)
            if len(item) == 1:
                item = item[0]
            yield self.collate_fn(item)
