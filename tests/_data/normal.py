# pylint: disable=missing-function-docstring
from typing import List
import torch


def sample_data(counts: List[int], dims: List[int]) -> List[torch.Tensor]:
    return [torch.randn(count, dim) for count, dim in zip(counts, dims)]


def sample_means(counts: List[int], dims: List[int]) -> List[torch.Tensor]:
    return [torch.randn(count, dim) for count, dim in zip(counts, dims)]


def sample_spherical_covars(counts: List[int]) -> List[torch.Tensor]:
    return [torch.rand(count) for count in counts]


def sample_diag_covars(counts: List[int], dims: List[int]) -> List[torch.Tensor]:
    return [torch.rand(count, dim).squeeze() for count, dim in zip(counts, dims)]


def sample_full_covars(counts: List[int], dims: List[int]) -> List[torch.Tensor]:
    result = []
    for count, dim in zip(counts, dims):
        A = torch.randn(count, dim * 10, dim)
        result.append(A.permute(0, 2, 1).bmm(A).squeeze())
    return result
