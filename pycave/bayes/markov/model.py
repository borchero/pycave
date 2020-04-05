import torch
import torch.nn as nn
import torch.distributions as dist
from torch.nn.utils.rnn import PackedSequence, pack_sequence
import pyblaze.nn as xnn
from pycave.bayes._internal.utils import power_iteration
from .config import MarkovModelConfig
from .engine import MarkovModelEngine

class MarkovModel(xnn.Estimator, xnn.Configurable, nn.Module):
    """
    The MarkovModel models a simple MarkovChain with a fixed set of states. You may use this class
    whenever states are known and transition probabilities are the only quantity of interest. In
    case of any additional output from the states, consider using the `HMM` model.
    """

    __config__ = MarkovModelConfig
    __engine__ = MarkovModelEngine

    def __init__(self, *args, **kwargs):
        """
        Initializes a new Markov model by passing a `MarkovModelConfig` or the config's
        configuration parameters as keyword arguments.
        """
        super().__init__(*args, **kwargs)

        self.initial_probs = nn.Parameter(
            torch.empty(self.num_states), requires_grad=False
        )
        self.transition_probs = nn.Parameter(
            torch.empty(self.num_states, self.num_states), requires_grad=False
        )

        self.reset_parameters()

    def reset_parameters(self):
        """
        Resets the parameter of the model by sampling initial probabilities as well as transition
        probabilities from a uniform distribution.
        """
        # 1) Initial Probabilities
        self.initial_probs.uniform_()
        self.initial_probs /= self.initial_probs.sum()

        # 2) Transition Probabilities
        self.transition_probs.uniform_()
        self.transition_probs /= self.transition_probs.sum(1, keepdim=True)

    def forward(self, data):
        """
        Runs inference for a single packed sequence, i.e. computes the negative log-likelihood of
        the given sequences.

        Parameters
        ----------
        data: torch.PackedSequence [N]
            The sequences for which to compute the negative log-likelihood (number of items N).

        Returns
        -------
        torch.Tensor [1]
            The negative log-likelihood.
        """
        num_sequences = data.batch_sizes[0].item()

        # 1) Get probabilities of first sequence values
        log_likeli = self.initial_probs[data.data[:num_sequences]].log()
        offset = num_sequences

        # 2) Iterate over transitions
        for prev_size, size in zip(data.batch_sizes, data.batch_sizes[1:]):
            source = data.data[offset-prev_size: offset-prev_size+size]
            target = data.data[offset: offset+size]
            log_likeli[:size] += self.transition_probs[source, target].log()
            offset += size

        # 3) Compute final negative log-likelihood
        return -log_likeli.logsumexp(-1).sum()

    def fit(self, *args, **kwargs):
        """
        Optimizes the model's parameters. Many parameters are not described here but can be inferred
        from `here <https://bit.ly/39FoKpe>`_. By default, data must be given as PyTorch data
        loader. However, in order to make calling this function easier for simple use cases, you
        can also supply the following instead:

        * A single packed sequence
        * A single 2-D tensor (interpreted as batch of sequences)
        * A list of packed sequences
        * A list of 2-D tensors (interpreted as batches of sequences)

        Apart from the data given as first parameter, the following keyword arguments may be given:

        Parameters
        ----------
        symmetric: bool, default: False
            Whether a symmetric transition matrix should be learnt from the data (e.g. useful
            when training on random walks from an undirected graph).
        teleport_alpha: float, default: 0
            The probability of random teleportations from one state to a randomly selected other
            one upon every transition. Generally "spaces out" probabilities in the transition
            probability matrix.

        Returns
        -------
        pyblaze.nn.History
            A history object carrying no information but the training time (`duration` attribute).
        """
        data = self._process_input(args[0])
        return super().fit(data, *args[1:], **kwargs)

    def evaluate(self, *args, **kwargs):
        """
        Computes the negative log-likelihood of the data under this model.

        Returns
        -------
        pyblaze.nn.Evaluation
            An evaluation object where the `neg_log_likelihood` property yields the per-datapiont
            negative log-likelihood.
        """
        data = self._process_input(args[0])
        return super().evaluate(data, *args[1:], **kwargs)

    def sample(self, num_sequences, sequence_length):
        """
        Samples the given number of sequences with the given length from the model's underlying
        probability distribution.

        Parameters
        ----------
        num_sequences: int
            The number of sequences to sample.
        sequence_length: int
            The length of the sequences to sample. Generation tends to be much slower for longer
            sequences compared to a higher number of sequences. The reason is that generation of
            sequences needs to be iterative.

        Returns
        -------
        torch.Tensor [N, S]
            The state sequences (number of sequences N, sequence length S).
        """
        samples = torch.empty(num_sequences, sequence_length, dtype=torch.long)

        # 1) Initialize initial states
        samples[:, 0] = self._sample_initial_states(num_sequences)

        # 2) Now sample the sequences
        for i in range(1, sequence_length):
            generator = dist.Categorical(self.transition_probs[samples[:, i-1]])
            samples[:, i] = generator.sample()

        return samples

    def stationary_distribution(self, max_iterations=100):
        """
        Computes the stationary distribution of the Markov chain. This equals the eigenvector
        corresponding to the largest eigenvalue of the transposed transition matrix.

        Parameters
        ----------
        max_iterations: int, default: 100
            The number of iterations to perform for the power iteration.

        Returns
        -------
        torch.Tensor [N]
            The probability of a random walker visiting each of the states after infinitely many
            steps.
        """
        return power_iteration(self.transition_probs.t(), max_iterations=max_iterations)

    def _sample_initial_states(self, num_samples):
        generator = dist.Categorical(self.initial_probs)
        return generator.sample((num_samples,))

    def _process_input(self, data):
        if isinstance(data, PackedSequence):
            return [data]
        if isinstance(data, torch.Tensor):
            return [pack_sequence(data)]
        if isinstance(data, (list, tuple)) and isinstance(data[0], torch.Tensor):
            return [pack_sequence(d) for d in data]
        return data

    def __repr__(self):
        return f'{self.__class__.__name__}(num_states={self.num_states})'
