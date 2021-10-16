from __future__ import annotations
from typing import Union
import numpy as np
import numpy.typing as npt
import torch
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

TabularData = Union[
    npt.NDArray[np.float32],
    torch.Tensor,
    Dataset[torch.Tensor],
]
TabularData.__doc__ = """
Data that may be passed to estimators expecting 2-D tabular data. Data may be provided in multiple
formats:

- NumPy array of shape ``[num_datapoints, dim]``.
- PyTorch tensor of shape ``[num_datapoints, dim]``.
- PyTorch dataset yielding items of shape ``[dim]``.
"""
