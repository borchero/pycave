from typing import Any, Callable, Optional
import torch
from torchmetrics import Metric
from pycave.bayes.core import covariance_shape, CovarianceType


class PriorAggregator(Metric):
    """
    The prior aggregator aggregates component probabilities over batches and process.
    """

    def __init__(
        self,
        num_components: int,
        *,
        dist_sync_fn: Optional[Callable[[Any], Any]] = None,
    ):
        super().__init__(dist_sync_fn=dist_sync_fn)  # type: ignore

        self.responsibilities: torch.Tensor
        self.add_state("responsibilities", torch.zeros(num_components), dist_reduce_fx="sum")

    def update(self, responsibilities: torch.Tensor) -> None:
        # Responsibilities have shape [N, K]
        self.responsibilities.add_(responsibilities.sum(0))

    def compute(self) -> torch.Tensor:
        return self.responsibilities / self.responsibilities.sum()


class MeanAggregator(Metric):
    """
    The mean aggregator aggregates component means over batches and processes.
    """

    def __init__(
        self,
        num_components: int,
        num_features: int,
        *,
        dist_sync_fn: Optional[Callable[[Any], Any]] = None,
    ):
        super().__init__(dist_sync_fn=dist_sync_fn)  # type: ignore

        self.mean_sum: torch.Tensor
        self.add_state("mean_sum", torch.zeros(num_components, num_features), dist_reduce_fx="sum")

        self.component_weights: torch.Tensor
        self.add_state("component_weights", torch.zeros(num_components), dist_reduce_fx="sum")

    def update(self, data: torch.Tensor, responsibilities: torch.Tensor) -> None:
        # Data has shape [N, D]
        # Responsibilities have shape [N, K]
        self.mean_sum.add_(responsibilities.t().matmul(data))
        self.component_weights.add_(responsibilities.sum(0))

    def compute(self) -> torch.Tensor:
        return self.mean_sum / self.component_weights.unsqueeze(1)


class CovarianceAggregator(Metric):
    """
    The covariance aggregator aggregates component covariances over batches and processes.
    """

    def __init__(
        self,
        num_components: int,
        num_features: int,
        covariance_type: CovarianceType,
        reg: float,
        *,
        dist_sync_fn: Optional[Callable[[Any], Any]] = None,
    ):
        super().__init__(dist_sync_fn=dist_sync_fn)  # type: ignore

        self.num_components = num_components
        self.num_features = num_features
        self.covariance_type = covariance_type
        self.reg = reg

        self.covariance_sum: torch.Tensor
        self.add_state(
            "covariance_sum",
            torch.zeros(covariance_shape(num_components, num_features, covariance_type)),
            dist_reduce_fx="sum",
        )

        self.component_weights: torch.Tensor
        self.add_state("component_weights", torch.zeros(num_components), dist_reduce_fx="sum")

    def update(
        self, data: torch.Tensor, responsibilities: torch.Tensor, means: torch.Tensor
    ) -> None:
        data_component_weights = responsibilities.sum(0)
        self.component_weights.add_(data_component_weights)

        if self.covariance_type in ("spherical", "diag"):
            x_prob = torch.matmul(responsibilities.t(), data.square())
            m_prob = data_component_weights.unsqueeze(-1) * means.square()
            xm_prob = means * torch.matmul(responsibilities.t(), data)
            covars = x_prob - 2 * xm_prob + m_prob
            if self.covariance_type == "diag":
                self.covariance_sum.add_(covars)
            else:  # covariance_type == "spherical"
                self.covariance_sum.add_(covars.mean(1))
        elif self.covariance_type == "tied":
            # This is taken from https://github.com/scikit-learn/scikit-learn/blob/
            # 844b4be24d20fc42cc13b957374c718956a0db39/sklearn/mixture/_gaussian_mixture.py#L183
            x_sq = data.T.matmul(data)
            mean_sq = (data_component_weights * means.T).matmul(means)
            self.covariance_sum.add_(x_sq - mean_sq)
        else:  # covariance_type == "full":
            # We iterate over each component since this is typically faster...
            for i in range(self.num_components):
                component_diff = data - means[i]
                covars = (responsibilities[:, i].unsqueeze(1) * component_diff).T.matmul(
                    component_diff
                )
                self.covariance_sum[i].add_(covars)

    def compute(self) -> torch.Tensor:
        if self.covariance_type == "diag":
            return self.covariance_sum / self.component_weights.unsqueeze(-1) + self.reg
        if self.covariance_type == "spherical":
            return self.covariance_sum / self.component_weights + self.reg * self.num_features
        if self.covariance_type == "tied":
            result = self.covariance_sum / self.component_weights.sum()
            result.flatten()[:: self.num_features + 1].add_(self.reg)
            return result
        # covariance_type == "full"
        result = self.covariance_sum / self.component_weights.unsqueeze(-1).unsqueeze(-1)
        diag_mask = (
            torch.eye(self.num_features, device=result.device, dtype=result.dtype)
            .bool()
            .unsqueeze(0)
            .expand(self.num_components, -1, -1)
        )
        result[diag_mask] += self.reg
        return result
