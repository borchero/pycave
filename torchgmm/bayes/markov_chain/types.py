from typing import List, Tuple, Union
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
SequenceData.__doc__ = """
Data that may be passed to estimators expecting 1-D sequences. Data may be provided in multiple
formats:

- NumPy array of shape ``[num_sequences, sequence_length]``.
- PyTorch tensor of shape ``[num_sequences, sequence_length]``.
- PyTorch dataset yielding items of shape ``[sequence_length]`` where the sequence length may
  differ for different indices.
"""


def collate_sequences_same_length(data: Tuple[torch.Tensor]) -> PackedSequence:
    """
    Collates the provided sequences into a packed sequence. Each sequence has to have the same
    length.

    Args:
        data: A single tensor of shape ``[num_sequences, sequence_length]`` where each item has the
            same length.

    Returns:
        A packed sequence containing all sequences.
    """
    (sequences,) = data
    num_sequences, sequence_length = sequences.size()
    batch_sizes = torch.ones(sequence_length, dtype=torch.long) * num_sequences
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
