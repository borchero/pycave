import math
from typing import cast, Dict, List, Optional
import pytorch_lightning as pl
import torch
from torch.nn.utils.rnn import PackedSequence
from .model import MarkovChainModel


class MarkovChainLightningModule(pl.LightningModule):
    """
    The lightning module...
    """

    def __init__(self, model: MarkovChainModel, symmetric: bool = False):
        super().__init__()
        self.automatic_optimization = False

        self.model = model
        self.symmetric = symmetric

    def configure_optimizers(self) -> None:
        return None

    def training_step(self, batch: PackedSequence, _batch_idx: int) -> Dict[str, torch.Tensor]:
        # Extract typed contents from packed sequence
        batch_sizes = cast(torch.Tensor, batch.batch_sizes)

        data = cast(torch.Tensor, batch.data)
        num_states = self.model.config.num_states
        num_sequences = batch_sizes[0].item()

        # Compute counts for initial states
        initial_counts = torch.bincount(data[:num_sequences], minlength=num_states).float()

        # Compute transition counts
        transition_counts = torch.zeros_like(self.model.transition_probs).view(-1)
        offset = 0
        for prev_size, size in zip(batch_sizes, batch_sizes[1:]):
            sources = data[offset : offset + size]
            targets = data[offset + prev_size : offset + prev_size + size]
            transitions = sources * num_states + targets
            values = torch.ones_like(transitions, dtype=torch.float)
            transition_counts.scatter_add_(0, transitions, values)
            offset += prev_size

        return {
            "initial_counts": initial_counts,
            "transition_counts": transition_counts,
        }

    def training_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]) -> None:
        # Aggregate outputs and make sure that the transition counts have a proper shape
        initial_counts = torch.stack([x["initial_counts"] for x in outputs]).sum(0)
        transition_counts = (
            torch.stack([x["transition_counts"] for x in outputs])
            .sum(0)
            .view_as(self.model.transition_probs)
        )

        # Then, check if we need a symmetric transition matrix
        if self.symmetric:
            transition_counts += transition_counts.t()

        # Eventually, assign to model buffers
        self.model.initial_probs.set_(_normalize(initial_counts))  # type: ignore
        self.model.transition_probs.set_(_normalize(transition_counts))  # type: ignore

    def test_step(self, batch: PackedSequence, _batch_idx: int) -> Dict[str, float]:
        out = self.model(batch)
        count = out.size(0)
        return {
            "mean": torch.logsumexp(out - math.log(count), 0).item(),  # type: ignore
            "count": count,
        }

    def test_epoch_end(self, outputs: List[Dict[str, float]]) -> None:
        counts = torch.as_tensor([x["count"] for x in outputs])
        weights = counts / counts.sum()
        means = torch.as_tensor([x["mean"] for x in outputs])
        self.log("log_prob", torch.logsumexp(means + weights.log(), 0).item())

    def predict_step(  # pylint: disable=signature-differs
        self, batch: PackedSequence, batch_idx: int, dataloader_idx: Optional[int]
    ) -> torch.Tensor:
        return self.model(batch)


def _normalize(tensor: torch.Tensor) -> torch.Tensor:
    return tensor / tensor.sum(-1, keepdim=True)
