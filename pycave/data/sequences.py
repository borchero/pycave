from typing import List, Union
import numpy as np
import torch
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence, pad_sequence
from torch.utils.data import Dataset

SequenceData = Union[np.ndarray, torch.Tensor, Dataset[torch.Tensor]]
"""
Data that may passed to estimators that work with sequence data. NumPy arrays and PyTorch tensors
must be two-dimensional with shape `[num_sequences, sequence_length]`. PyTorch datasets must yield
individual one-dimensional sequences (which may have different length).
"""


def sequence_dataset_from_data(data: SequenceData) -> Dataset[torch.Tensor]:
    """
    Generates a PyTorch dataset for the provided sequence data.

    Args:
        data: The sequence data.

    Returns:
        A PyTorch dataset yielding one-dimensional sequences, potentially of differing size.
    """
    if isinstance(data, np.ndarray):
        return sequence_dataset_from_data(torch.from_numpy(data))
    if isinstance(data, torch.Tensor):
        assert data.dim() == 2, "data given as array or tensor must be two-dimensional"
        return _OneDimTensorDataset(data)
    return data


def collate_sequences(sequences: List[torch.Tensor]) -> PackedSequence:
    """
    Collates the sequences provided as a list into a packed sequence. The sequences are not
    required to be sorted.

    Args:
        sequences: A list of one-dimensional tensors to batch.

    Returns:
        A packed sequence with all the data provided.
    """
    return pack_padded_sequence(
        pad_sequence(sequences, batch_first=False),
        lengths=torch.as_tensor([t.size(0) for t in sequences]),
        batch_first=False,
        enforce_sorted=False,
    )


class _OneDimTensorDataset(Dataset[torch.Tensor]):
    def __init__(self, data: torch.Tensor):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.data[index]
