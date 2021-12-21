# pyright: reportPrivateUsage=false, reportUnknownParameterType=false
from dataclasses import dataclass
from typing import overload
import torch
import torch._jit_internal as _jit
from lightkit.nn import Configurable
from torch import jit, nn
from torch.nn.utils.rnn import PackedSequence


@dataclass
class MarkovChainModelConfig:
    """
    Configuration class for a Markov chain model.

    See also:
        :class:`MarkovChainModel`
    """

    #: The number of states that are managed by the Markov chain.
    num_states: int


class MarkovChainModel(Configurable[MarkovChainModelConfig], nn.Module):
    """
    PyTorch module for a Markov chain. The initial state probabilities as well as the transition
    probabilities are non-trainable parameters.
    """

    def __init__(self, config: MarkovChainModelConfig):
        """
        Args:
            config: The configuration to use for initializing the module's buffers.
        """
        super().__init__(config)

        #: The probabilities for the initial states, buffer of shape ``[num_states]``.
        self.initial_probs: torch.Tensor
        self.register_buffer("initial_probs", torch.empty(config.num_states))

        #: The transition probabilities between all states, buffer of shape
        #: ``[num_states, num_states]``.
        self.transition_probs: torch.Tensor
        self.register_buffer("transition_probs", torch.empty(config.num_states, config.num_states))

        self.reset_parameters()

    @jit.unused
    def reset_parameters(self) -> None:
        """
        Resets the parameters of the Markov model. Initial and transition probabilities are sampled
        uniformly.
        """
        nn.init.uniform_(self.initial_probs)
        self.initial_probs.div_(self.initial_probs.sum())

        nn.init.uniform_(self.transition_probs)
        self.transition_probs.div_(self.transition_probs.sum(1, keepdim=True))

    @overload
    @_jit._overload_method  # pylint: disable=protected-access
    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        ...

    @overload
    @_jit._overload_method  # pylint: disable=protected-access
    def forward(self, sequences: PackedSequence) -> torch.Tensor:
        ...

    def forward(self, sequences) -> torch.Tensor:  # type: ignore
        """
        Computes the log-probability of observing each of the provided sequences.

        Args:
            sequences: Tensor of shape ``[num_sequences, sequence_length]`` or a packed sequence.
                Packed sequences should be used whenever the sequence lengths differ. All
                sequences must contain state indices of dtype ``long``.

        Returns:
            A tensor of shape ``[sequence_length]``, returning the log-probability of each
                sequence.
        """
        if isinstance(sequences, torch.Tensor):
            log_probs = self.initial_probs[sequences[:, 0]].log()
            sources = sequences[:, :-1]
            targets = sequences[:, 1:].unsqueeze(-1)
            transition_probs = self.transition_probs[sources].gather(-1, targets).squeeze(-1)
            return log_probs + transition_probs.log().sum(-1)
        if isinstance(sequences, PackedSequence):
            data = sequences.data
            batch_sizes = sequences.batch_sizes

            log_probs = self.initial_probs[data[: batch_sizes[0]]].log()
            offset = 0
            for prev_size, curr_size in zip(batch_sizes, batch_sizes[1:]):
                log_probs[:curr_size] += self.transition_probs[
                    data[offset : offset + curr_size],
                    data[offset + prev_size : offset + prev_size + curr_size],
                ].log()
                offset += prev_size

            if sequences.unsorted_indices is not None:
                return log_probs[sequences.unsorted_indices]
            return log_probs
        raise ValueError("unsupported input type")

    def sample(self, num_sequences: int, sequence_length: int) -> torch.Tensor:
        """
        Samples random sequences from the Markov chain.

        Args:
            num_sequences: The number of sequences to sample.
            sequence_length: The length of all sequences to sample.

        Returns:
            Tensor of shape ``[num_sequences, sequence_length]`` with dtype ``long``, providing the
            sampled states.
        """
        samples = torch.empty(
            num_sequences, sequence_length, device=self.transition_probs.device, dtype=torch.long
        )
        samples[:, 0] = self.initial_probs.multinomial(num_sequences, replacement=True)
        for i in range(1, sequence_length):
            samples[:, i] = self.transition_probs[samples[:, i - 1]].multinomial(1).squeeze(-1)
        return samples

    def stationary_distribution(
        self, tol: float = 1e-7, max_iterations: int = 1000
    ) -> torch.Tensor:
        """
        Computes the stationary distribution of the Markov chain using power iteration.

        Args:
            tol: The tolerance to use when checking if the power iteration has converged. As soon
                as the norm between the vectors of two successive iterations is below this value,
                the iteration is stopped.
            max_iterations: The maximum number of iterations to run if the tolerance does not
                indicate convergence.

        Returns:
            A tensor of shape ``[num_states]`` with the stationary distribution (i.e. the
                eigenvector corresponding to the largest eigenvector of the transition matrix,
                normalized to describe a probability distribution).
        """
        A = self.transition_probs.t()
        v = torch.rand(A.size(0), device=A.device, dtype=A.dtype)

        for _ in range(max_iterations):
            v_old = v
            v = A.mv(v)
            v = v / v.norm()
            if (v - v_old).norm() < tol:
                break

        return v / v.sum()
