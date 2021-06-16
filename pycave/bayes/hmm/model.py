import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pack_sequence
import pyblaze.nn as xnn
from pycave.bayes.markov.model import MarkovModel
from pycave.bayes._internal.output import Discrete, Gaussian
from pycave.bayes._internal.utils import normalize, packed_get_last
from .engine import HMMEngine

class HMM(xnn.Estimator, nn.Module):
    """
    The HMM represents a hidden Markov model with different kinds of emissions.

    In addition to the methods documented below, the HMM provides the following methods as provided
    by the `estimator mixin <https://bit.ly/2wiUB1i>`_.

    `fit(...)`
        Optimizes the model's parameters.

    `evaluate(...)`
        Computes the per-datapoint negative log-likelihood of the given data.

    `predict(...)`
        Performs filtering or smoothing based on the passed parameters, returning the distribution
        over hidden states at the last timestep of each sequence or at every timestep, respectively.

    The parameters that may be passed to the functions can be derived from the
    `engine documentation <https://bit.ly/3bYHhOV>`_. The data needs, however, not be passed as a
    PyTorch data loader but all methods also accept the following instead:

    * A single packed sequence
    * A single 2-D tensor (interpreted as batch of sequences)
    * A list of packed sequences
    * A list of 2-D tensors (interpreted as batches of sequences)

    Additionally, the methods allow for the following keyword arguments:

    `fit(...)`
        * epochs: int, default: 20
            The maximum number of iterations to run training for.
        * eps: float, default: 0.01
            The minimum per-datapoint difference in the negative log-likelihood to consider a
            model "better".
        * patience: int, default: 0
            The number of times the negative log-likelihood may be above the minimum that has
            already been achieved before aborting training.

    `predict(...)`
        * smooth: bool, default: False
            Whether to perform smoothing and return distributions over hidden states for all time
            steps of the sequences.
    """

    @property
    def engine(self):
        return HMMEngine(self)

    def __init__(self, num_states, output='gaussian', output_num_states=1, output_dim=1,
                 output_covariance='diag'):
        """
        Initialies a new HMM.

        Parameters
        ----------
        num_states: int
            The number of states in the hidden Markov model.
        output: str, default: 'gaussian'
            The type of output that the HMM generates. Currently must be one of the following:

                * gaussian: Every state's output is a Gaussian distribution with some mean and
                    covariance.
                * discrete: All states have a shared set of discrete output states.
        output_num_states: int, default: 1
            The number of output states. Only applies if the output is discrete. Should be given in
            this case.
        output_dim: int, default: 1
            The dimensionality of the gaussian distributions. Only applies if the output is
            Gaussian. Should be given in this case.
        output_covariance: str, default: 'diag'
            The type of covariance to use for the gaussian distributions. The same constraints as
            for the Gaussian mixture model apply. Only applies if the output is Gaussian.
        """
        super().__init__()

        if output not in ('gaussian', 'discrete'):
            raise ValueError(f"Invalid output type '{output}'.")

        self.num_states = num_states
        self.output = output
        self.output_num_states = output_num_states
        self.output_dim = output_dim
        self.output_covariance = output_covariance

        self.markov = MarkovModel(num_states=self.num_states)

        if self.output == 'gaussian':
            self.emission = Gaussian(self.num_states, self.output_dim, self.output_covariance)
        elif self.output == 'discrete':
            self.emission = Discrete(self.num_states, self.output_num_states)

        self.reset_parameters()

    def reset_parameters(self, data=None, max_iter=100):
        """
        Resets the parameters of the model. The initial probabilities as well as the transition
        probabilities are initialized by drawing from the uniform distribution

        Depending on the output type, additional setup can be performed. If the output is Gaussian,
        an initial guess on the means of the state outputs via K-Means clustering can be made.
        Otherwise they are set randomly to values drawn from the (multivariate) standard normal
        distribution. The covariances are initialized according to the standard normal distribution.

        Parameters
        ----------
        data: torch.Tensor [N, D]
            (A subset of) datapoints to initialize the output parameters of this model (must not
            be shaped as a sequence). Ensure that this tensor is not too large as K-Means will run
            forever otherwise (number of datapoints N, dimensionality D).
        """
        self.markov.reset_parameters()
        self.emission.reset_parameters(data=data, max_iter=max_iter)

    def forward(self, data, smooth=False, return_emission=False):
        """
        Runs inference (filtering/smoothing) for a single packed sequence.

        Parameters
        ----------
        data: torch.PackedSequence [N]
            The sequences for which to compute alpha/beta values (number of items N).
        smooth: bool, default: False
            Whether to perform filtering or smoothing.
        return_emission: bool, default: False

        Returns
        -------
        torch.Tensor [N, K]
            The emission probabilities for all datapoints if `return_emission` is `True`.
        torch.Tensor ([S, K] or [N, K])
            The (normalized) alpha values, either for each sequence or for all timesteps of each
            sequence if `smooth` is `True`. The alpha values represent the filtered distribution
            over hidden states (number of sequences S, number of hidden states K).
        torch.Tensor [N, K]
            The (normalized) beta values if `smooth` is `True`.
        torch.Tensor [1]
            The negative log-likelihood of the data under the model (not normalized for the number
            of datapoints).
        """
        emission_probs = self.emission.evaluate(data.data)

        if not smooth:
            result = self._forward(data.batch_sizes, emission_probs)
        else:
            result = self._forward_backward(data.batch_sizes, emission_probs)

        if return_emission:
            return (emission_probs,) + result
        return result

    def sample(self, num_sequences, sequence_length):
        """
        Samples the specified number of sequences of the specified length from the hidden Markov
        model.

        Parameters
        ----------
        num_sequences: int
            The number of sequences to generate.
        sequence_length: int
            The length of the sequences to generate. Generation of the hidden states is done via
            `MarkovModel.sample`. Read its documentation about the performance of a long sequence
            length.

        Returns
        -------
        torch.Tensor [N, S, ?]
            Returns the sampled output whose shape depends on the type of output (number of
            sequences N, sequence length S):

            * gaussian: `torch.Tensor [N, S, D]` (dimensionality of Gaussians D).
            * discrete: `torch.Tensor [N, S]` where the values indicate the output state.
        """
        states = self.markov.sample(num_sequences, sequence_length)
        return self.emission.sample(states)

    #########################
    ### FORWARD ALGORITHM ###
    #########################
    def _forward(self, batch_sizes, emission_probs):
        num_sequences = batch_sizes[0]
        max_length = batch_sizes.size(0)
        data_offset = 0

        log_denom = torch.zeros(num_sequences, 1, device=emission_probs.device)

        # 1) Initialize alpha if not given
        alpha = torch.empty(num_sequences, self.num_states, device=emission_probs.device)
        alpha, denom = normalize(
            self.markov.initial_probs * emission_probs[:num_sequences],
            return_denom=True
        )
        data_offset += num_sequences
        log_denom += denom.log()

        # 2) Run forward pass for emissions
        for t in range(max_length - 1):
            bs = batch_sizes[t+1]
            probs = emission_probs[data_offset: data_offset+bs]
            alpha[:bs], denom = normalize(
                probs * (alpha[:bs] @ self.markov.transition_probs),
                return_denom=True
            )
            data_offset += bs
            log_denom[:bs] += denom.log()

        # 3) Compute log-likelihood
        log_likeli = (alpha.log() + log_denom).logsumexp(-1).sum()

        return alpha, -log_likeli

    ##################################
    ### FORWARD-BACKWARD ALGORITHM ###
    ##################################
    def _forward_backward(self, batch_sizes, emission_probs):
        M = self.num_states
        device = emission_probs.device
        max_length = batch_sizes.size(0)

        # 1) Initialize (empty) parameters
        alpha = torch.empty(batch_sizes.sum(), M, device=device)
        beta = torch.empty_like(alpha)
        alpha_log_denom = torch.zeros(batch_sizes[0], 1, device=device)

        # 2) Run forward phase
        data_offset = batch_sizes[0].item()
        alpha[:data_offset], denom = normalize(
            self.markov.initial_probs * emission_probs[:data_offset],
            return_denom=True
        )
        alpha_log_denom += denom.log()

        for t in range(max_length - 1):
            prev_bs = batch_sizes[t].item()
            bs = batch_sizes[t+1].item()

            probs = emission_probs[data_offset: data_offset+bs]
            previous = alpha[data_offset-prev_bs: data_offset-prev_bs+bs]

            alpha[data_offset: data_offset+bs], denom = normalize(
                probs * (previous @ self.markov.transition_probs),
                return_denom=True
            )

            alpha_log_denom[:bs] += denom.log()
            data_offset += bs

        # 3) Run backward phase (data_offset is sum of all batch sizes here)
        batch_sum = data_offset

        # 3.1) First, set all ones at the end of sequences
        for t in range(max_length - 1, -1, -1):
            prev_bs = batch_sizes[t+1].item() if t < max_length - 1 else 0
            bs = batch_sizes[t].item()
            num_endings = bs - prev_bs

            beta[data_offset-num_endings: data_offset] = 1
            data_offset -= bs

        # 3.2) Now, iterate
        data_offset = batch_sum - batch_sizes[max_length - 1].item()
        for t in range(max_length - 1, 0, -1):
            # batch size of subsequent timestep - these need to be updated
            bs = batch_sizes[t].item()
            # this is the current timestep - only for `bs` need update
            current_bs = batch_sizes[t-1].item()

            probs = emission_probs[data_offset: data_offset+bs]
            prev_beta = beta[data_offset: data_offset+bs]

            beta[data_offset-current_bs: data_offset-current_bs+bs] = \
                normalize((probs * prev_beta) @ self.markov.transition_probs.t())

            data_offset -= current_bs

        # 4) Compute log-likelihood
        alpha_ = packed_get_last(alpha, batch_sizes)
        log_likeli = (alpha_.log() + alpha_log_denom).logsumexp(-1).sum()

        return alpha, beta, -log_likeli

    #############
    ### UTILS ###
    #############
    def prepare_input(self, data):
        if isinstance(data, PackedSequence):
            return [data]
        if isinstance(data, torch.Tensor):
            return [pack_sequence(data)]
        if isinstance(data, (list, tuple)) and isinstance(data[0], torch.Tensor):
            return [pack_sequence(d) for d in data]
        return data
