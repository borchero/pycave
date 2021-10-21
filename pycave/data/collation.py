from typing import List
import torch
from torch.nn.utils.rnn import pack_sequence, PackedSequence


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
