from typing import Any, Callable, cast, Optional, Tuple
import torch
from torch.nn.utils.rnn import PackedSequence
from torchmetrics import Metric


class StateCountAggregator(Metric):
    """
    The state count aggregator aggregates initial states and transitions between states.
    """

    def __init__(
        self,
        num_states: int,
        symmetric: bool,
        *,
        dist_sync_fn: Optional[Callable[[Any], Any]] = None
    ):
        super().__init__(dist_sync_fn=dist_sync_fn)  # type: ignore

        self.num_states = num_states
        self.symmetric = symmetric

        self.initial_counts: torch.Tensor
        self.add_state("initial_counts", torch.zeros(num_states), dist_reduce_fx="sum")

        self.transition_counts: torch.Tensor
        self.add_state(
            "transition_counts", torch.zeros(num_states, num_states).view(-1), dist_reduce_fx="sum"
        )

    def update(self, sequences: PackedSequence) -> None:
        batch_sizes = cast(torch.Tensor, sequences.batch_sizes)
        num_sequences = batch_sizes[0]
        data = cast(torch.Tensor, sequences.data)

        # First, we count the initial states
        initial_counts = torch.bincount(data[:num_sequences], minlength=self.num_states).float()
        self.initial_counts.add_(initial_counts)

        # Then, we count the transitions
        offset = 0
        for prev_size, size in zip(batch_sizes, batch_sizes[1:]):
            sources = data[offset : offset + size]
            targets = data[offset + prev_size : offset + prev_size + size]
            transitions = sources * self.num_states + targets
            values = torch.ones_like(transitions, dtype=torch.float)
            self.transition_counts.scatter_add_(0, transitions, values)
            offset += prev_size

    def compute(self) -> Tuple[torch.Tensor, torch.Tensor]:
        initial_probs = self.initial_counts / self.initial_counts.sum()
        transition_counts = self.transition_counts.view(self.num_states, self.num_states)

        if self.symmetric:
            self.transition_counts.add_(transition_counts.t())
        transition_probs = transition_counts / transition_counts.sum(1, keepdim=True)

        return initial_probs, transition_probs
