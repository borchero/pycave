import time
import torch
import torch.nn as nn
import torch.distributions as dist
import bxtorch.nn as xnn
from .utils import power_iteration

class MarkovModelConfig(xnn.Config):
    """
    The Markov model config can be used to customize the Markov model.
    """

    num_states: int
    """
    The number of states in the Markov model.
    """

    symmetric: bool = False
    """
    Whether to use a symmetric transition probability matrix.
    """


# pylint: disable=abstract-method
class MarkovModel(xnn.Configurable, xnn.Estimator, nn.Module):
    """
    The MarkovModel models a simple MarkovChain with a fixed set of states. You may use this class
    whenever states are known and transition probabilities are the only quantity of interest. In
    case of any additional output from the states, consider using the `HMM` model.
    """

    __config__ = MarkovModelConfig

    # MARK: Initialization
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

    # MARK: Instance Methods
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

    def fit(self, sequences, teleport_alpha=0):
        """
        Optimizes the parameters of the markov model from the given sequences.

        Parameters
        ----------
        sequences: torch.Tensor [N, S]
            N sequences to train the Markov model on. The sequences are expected to be of uniform
            length S. Each value of the tensor must be the index of a state.
        teleport_alpha: float, default: 0
            The probability of "invalid" transitions in the given sequences. This setting is
            motivated by a random walker which teleports after every step with probability alpha.

        Returns
        -------
        bxtorch.nn.History
            For completeness, it returns a history object. However, apart from the duration of the
            training, the object does not carry any information.
        """
        tic = time.time()

        # 1) Extract information
        initial_states = sequences[:, 0]
        transitions = _get_transitions(sequences, self.symmetric)

        # 2) Update initial probabilities
        counts = torch.bincount(initial_states, minlength=self.num_states)
        counts = counts.float()
        self.initial_probs.set_(counts / counts.sum())

        # 3) Update transition probabilities
        tr_counts = _count_transitions(transitions, self.num_states)
        tr_matrix = tr_counts / tr_counts.sum(1, keepdim=True)

        if teleport_alpha > 0:
            teleport_factor = teleport_alpha / tr_matrix.size(0)
            teleport_matrix = torch.ones_like(tr_matrix)
            beta = 1 - teleport_alpha
            tr_matrix = (tr_matrix - teleport_factor * teleport_matrix) / beta

        self.transition_probs.set_(tr_matrix)

        return xnn.History(time.time() - tic, [])

    def evaluate(self, sequences):
        """
        Computes the negative log-likelihood of observing the given sequences under this stochastic
        model.

        Parameters
        ----------
        sequences: torch.Tensor [N, S]
            N sequences of S states.

        Returns
        -------
        float
            The negative log-likelihood divided by the number of sequences.
        """
        initial_log_probs = self.initial_probs[sequences[:, 0]].log()
        transitions = _get_transitions(sequences)
        transition_log_probs = self.transition_probs[transitions].log()
        log_prob = initial_log_probs.sum() + transition_log_probs.sum()
        return log_prob / sequences.size(0)

    def predict(self):
        """
        Only implemented for completeness, does not do anything.
        """
        raise AttributeError(
            f"{self.__class__.__name__} does not provide a predict method."
        )

    def sample(self, num_sequences, sequence_length):
        """
        Samples the given number of sequences with the given length from the model's underlying
        probability distribution.

        Parameters
        ----------
        num_sequences: int
            The number N of sequences to sample.
        sequence_length: int
            The length S of the sequences to sample.

        Returns
        -------
        torch.Tensor [N, S]
            The N state sequences of length S.
        """
        samples = torch.empty(num_sequences, sequence_length, dtype=torch.long)

        # 1) Initialize initial states
        samples[:, 0] = dist.Categorical(self.initial_probs).sample(
            (num_sequences,)
        )

        # 2) Now sample the sequences
        for i in range(1, sequence_length):
            samples[:, i] = dist.Categorical(
                self.transition_probs[samples[:, i-1]]
            ).sample()

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


def _get_transitions(sequences, symmetric=False):
    # pylint: disable=not-callable
    transitions = torch.tensor(
        list(zip(sequences[:, :-1].tolist(), sequences[:, 1:].tolist())),
        device=sequences.device
    ).permute(0, 2, 1).contiguous().view(-1, 2)

    if symmetric:
        transitions = torch.cat([
            transitions,
            transitions.roll(1, dims=1)
        ])

    return transitions


def _count_transitions(transitions, num_states):
    return torch.sparse.FloatTensor(
        transitions.t(), torch.ones(transitions.size(0)),
        (num_states, num_states),
        device=transitions.device
    ).to_dense()
