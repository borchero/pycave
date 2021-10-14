from __future__ import annotations
from typing import List, Union
import numpy as np
import numpy.typing as npt
import torch
from torch.nn.utils.rnn import pack_sequence, PackedSequence
from torch.utils.data import Dataset

SequenceData = Union[
    npt.NDArray[np.float32],
    torch.Tensor,
    Dataset[torch.Tensor],
]
"""
Data that may passed to estimators that work with sequence data. NumPy arrays and PyTorch tensors
must be at least two-dimensional with shape `[num_sequences, sequence_length, ...]`. PyTorch
datasets must yield individual one- or higher-dimensional sequences (which may have different
length).

- NumPy array of shape ``[num_sequences, sequence_length]``.
- PyTorch tensor of shape ``[num_sequences, sequence_length]``.
- PyTorch dataset yielding items of shape ``[sequence_length]`` where the sequence length may
    differ for different indices.
"""

TabularData = Union[
    npt.NDArray[np.float32],
    torch.Tensor,
    Dataset[torch.Tensor],
]
"""
Data that may be passed to estimators expecting 2-D tabular data. Data may be provided in multiple
formats:

- NumPy array of shape ``[num_datapoints, dim]``.
- PyTorch tensor of shape ``[num_datapoints, dim]``.
- PyTorch dataset yielding items of shape ``[dim]``.
"""


def collate_sequences_same_length(sequences: torch.Tensor) -> PackedSequence:
    """
    Collates the provided sequences into a packed sequence. Each sequence has to have the same
    length.

    Args:
        sequences: A tensor of shape ``[num_sequences, sequence_length]`` where each item has the
            same length.

    Returns:
        A packed sequence containing all sequences.
    """
    num_sequences, sequence_length = sequences.size()
    batch_sizes = torch.ones(sequence_length) * num_sequences
    return PackedSequence(sequences.t().flatten(), batch_sizes)


def collate_sequences(sequences: List[torch.Tensor]) -> PackedSequence:
    """
    Collates the sequences provided as a list into a packed sequence. The sequences are not
    required to be sorted by their lengths.

    Args:
        sequences: A list of one-dimensional tensors to batch.

    Returns:
        A packed sequence with all the data provided.
    """
    return pack_sequence(sequences, enforce_sorted=False)
