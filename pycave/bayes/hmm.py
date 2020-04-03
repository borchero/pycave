import time
import torch
import torch.nn as nn
import pyblaze.nn as xnn
from pyblaze.utils.stdio import ProgressBar
from .markov import MarkovModel
from .output import Discrete, Gaussian
from .utils import batch_weights, normalize, packed_drop_last, packed_get_last

class HMMConfig(xnn.Config):
    """
    The configuration can be used to customize a Gaussian hidden Markov model. Values that do not
    have a default value must be passed to the initializer of the GMM.
    """

    num_states: int
    """
    The number of states in the hidden Markov model.
    """

    symmetric: bool = False
    """
    Whether to use a symmetric transition probability matrix.
    """

    output: str = 'gaussian'
    """
    The type of output that the HMM generates. Currently must be one of the following:

    * gaussian: Every state's output is a Gaussian distribution with some mean and covariance.
    * discrete: All states have a shared set of discrete output states.
    """

    output_num_states: int = 1
    """
    The number of output states. Only applies if the output is discrete. Should be given in this
    case.
    """

    output_dim: int = 1
    """
    The dimensionality of the gaussian distributions. Only applies if the output is Gaussian. Should
    be given in this case.
    """

    output_covariance: str = 'diag'
    """
    The type of covariance to use for the gaussian distributions. The same constraints as for the
    Gaussian mixture model apply (see `GMMConfig`). Only applies if the output is Gaussian.
    """

    def is_valid(self):
        return (
            self.output in ('gaussian', 'discrete') and
            self.output_covariance in ('diag', 'spherical', 'diag-shared')
        )

# pylint: disable=abstract-method
class HMM(xnn.Configurable, nn.Module):
    """
    The HMM represents a hidden Markov model with different kinds of emissions.
    """

    __config__ = HMMConfig

    def __init__(self, *args, **kwargs):
        """
        Initialies a new HMM either with a given `HMMConfig` or with the config's parameters passed
        as keyword arguments.
        """
        super().__init__(*args, **kwargs)

        self.markov = MarkovModel(
            num_states=self.num_states,
            symmetric=self.symmetric
        )

        if self.output == 'gaussian':
            self.emission = Gaussian(self.num_states, self.output_dim, self.output_covariance)
        elif self.output == 'discrete':
            self.emission = Discrete(self.num_states, self.output_num_states)

        self.reset_parameters()

    def reset_parameters(self, sequences=None, max_iter=100):
        """
        Resets the parameters of the model. The initial probabilities as well as the transition
        probabilities are initialized by drawing from the uniform distribution

        Depending on the output type, additional setup can be performed. If the output is Gaussian,
        an initial guess on the means of the state outputs via K-Means clustering can be made.
        Otherwise they are set randomly to values drawn from the (multivariate) standard normal
        distribution. The covariances are initialized according to the standard normal distribution.

        Parameters
        ----------
        sequences: torch.Tensor [N, S, K]
            (A subset of) sequences to initialize the output parameters of this model.
        """
        self.markov.reset_parameters()

        if sequences is not None:
            sequences = sequences.view(-1, sequences.size(-1))
        self.emission.reset_parameters(data=sequences, max_iter=max_iter)

    def fit(self, sequences, max_iter=100, batch_size=None, eps=1e-4, verbose=False):
        """
        Optimizes the HMM's parameters according to the given sequences. Consider running
        `reset_parameters` on (a subset of) the sequences first to achieve better results if output
        is Gaussian.

        Parameters
        ----------
        sequences: torch.Tensor [N, S, ?]
            The data to train the model on where the shape depends on the type of output (number of
            sequences N, sequence length S):

            * gaussian: `torch.Tensor [N, S, D]` (emission dimensionality D).
            * discrete: `torch.Tensor [N, S]` where integer values indicate output states.
        max_iter: int, default: 100
            The maximum number of iterations to run the Baum-Welch algorithm for.
        batch_size: int, default: None
            The batch size to use for traning. If `None` is given, the full data is used.
        eps: float, default: 1e-4
            The difference in the log-likelihood (per datapoint) such that convergence is
            indicated.
        verbose: bool, default: False
            Whether to show a progress bar during training.

        Returns
        -------
        pyblaze.nn.History
            A history object containing the negative log-likelihoods over the course of the
            training (attribute `neg_log_likelihood`).
        """
        history = []
        tic = time.time()

        # 1) Initialize batch weights and set up early stopping
        # num_batches, weights = batch_weights(sequences.size(0), batch_size or sequences.size(0))
        num_batches = 1
        best_neg_log_likeli = float('inf')

        # 2) Run training
        for _ in ProgressBar(max_iter, verbose=verbose):
            if num_batches == 1:
                if isinstance(sequences, nn.utils.rnn.PackedSequence):
                    updates = self._packed_baum_welch_full(sequences)
                else:
                    updates = self._baum_welch_full(sequences)
            else:
                pass
                # updates = self._baum_welch_batch(sequences, batch_size, weights)

            # 2.1) Update parameters
            self.markov.initial_probs.set_(updates['initial_probs'])
            self.markov.transition_probs.set_(updates['transition_probs'])
            self.emission.apply(updates['output_update'])

            # 2.2) Update history
            nll = updates['neg_log_likelihood']
            history.append({'neg_log_likelihood': nll})

            # 2.3) Check for convergence
            if best_neg_log_likeli - nll < eps:
                break
            best_neg_log_likeli = nll

        return xnn.History(time.time() - tic, history)

    def predict(self, sequences, smooth=False, initial=None):
        """
        Predicts a probability distribution over the hidden states of the HMM based on the given
        sequences.

        Parameters
        ----------
        sequences: torch.Tensor [N, S, ?] or torch.PackedSequence
            The data to make predictions for. If all sequences to make predictions for have the same
            length, the may be given as simple tensor. Here, the shape depends on the type of
            output. Otherwise, you should provide a packed sequence (e.g. via
            `torch.nn.utils.rnn.pack_sequence`).
        smooth: bool, default: False
            Defines if filtering or smoothing should be performed. If this is set to `False`, only
            the probability distribution for the last step of the sequence will be returned.
            Otherwise, probability distributions for all steps will be returned. Running an argmax
            yields the sequence of most likely states (which is NOT equal to the most likely
            sequence of states). If you have no reason to set this value to `True`, don't do so as
            memory requirements increase linearly with the sequence length if this value is
            `False`.
        initial: torch.Tensor [N, K], default: None
            The initial distribution of hidden states. Must only be given if `smooth` is `False` and
            enables running the filtering algorithm in an online manner. Must be normalized in the
            last dimension (number of hidden variables K).

        Returns
        -------
        torch.Tensor ([N, K] or [N, S, K]) or list of torch.Tensor [S, K]
            Distribution over hidden variables, either for all sequence steps or only for the last,
            depending on the value of the `smooth` parameter. The distribution is normalized in the
            last dimension (number of hidden variables K).
        """
        if isinstance(sequences, torch.nn.utils.rnn.PackedSequence):
            return self._packed_predict(sequences, smooth, initial)
        return self._batch_predict(sequences, smooth, initial)

    def _batch_predict(self, sequences, smooth, initial):
        N = sequences.size(0)
        S = sequences.size(1)

        emission_probs = self.emission.evaluate(sequences)

        if smooth:
            alpha, beta, _ = self._forward_backward(N, S, emission_probs)
            return normalize(alpha * beta)

        return self._forward(N, S, emission_probs, initial)

    def _packed_predict(self, sequences, smooth, initial):
        emission_probs = self.emission.evaluate(sequences.data)

        if smooth:
            alpha, beta, _ = \
                self._packed_forward_backward(sequences.batch_sizes, emission_probs)
            gamma = normalize(alpha * beta)
            packed = nn.utils.rnn.PackedSequence(data=gamma, batch_sizes=sequences.batch_sizes)
            padded, lengths = nn.utils.rnn.pad_packed_sequence(packed, batch_first=True)

            if sequences.unsorted_indices is not None:
                return [padded[i, :lengths[i]] for i in sequences.unsorted_indices]
            return [padded[i, :lengths[i]] for i in range(lengths.size(0))]

        result = self._packed_forward(sequences.batch_sizes, emission_probs, initial)
        if sequences.unsorted_indices is not None:
            return result[sequences.unsorted_indices]
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
    def _forward(self, num_sequences, sequence_length, emission_probs, alpha):
        off = 1 if alpha is None else 0

        # 1) Initialize alpha if not given
        if alpha is None:
            alpha = torch.empty(num_sequences, self.num_states, device=emission_probs.device)
            alpha = normalize(self.markov.initial_probs * emission_probs[:, 0])

        # 2) Run forward pass for emissions
        for t in range(sequence_length - off):
            alpha = normalize(emission_probs[:, t+off] * (alpha @ self.markov.transition_probs))

        return alpha

    def _packed_forward(self, batch_sizes, emission_probs, alpha):
        off = 1 if alpha is None else 0
        num_sequences = batch_sizes[0]
        max_length = batch_sizes.size(0)
        data_offset = 0

        # 1) Initialize alpha if not given
        if alpha is None:
            alpha = torch.empty(num_sequences, self.num_states, device=emission_probs.device)
            alpha = normalize(self.markov.initial_probs * emission_probs[:batch_sizes[0]])
            data_offset += batch_sizes[0]

        # 2) Run forward pass for emissions
        for t in range(max_length - off):
            bs = batch_sizes[t + off]
            probs = emission_probs[data_offset: data_offset+bs]
            alpha[:bs] = normalize(probs * (alpha[:bs] @ self.markov.transition_probs))
            data_offset += bs

        return alpha

    ##################################
    ### FORWARD-BACKWARD ALGORITHM ###
    ##################################
    def _forward_backward(self, num_sequences, sequence_length, emission_probs):
        M = self.num_states
        device = emission_probs.device

        # 1) Initialize (empty) parameters
        alpha = torch.empty(num_sequences, sequence_length, M, device=device)
        beta = torch.empty_like(alpha)
        alpha_log_denom = torch.zeros(num_sequences, 1, device=device)

        # 2) Run forward phase
        alpha[:, 0] = normalize(self.markov.initial_probs * emission_probs[:, 0])
        for t in range(sequence_length - 1):
            alpha[:, t+1], denom = normalize(
                emission_probs[:, t+1] * (alpha[:, t] @ self.markov.transition_probs),
                return_denom=True
            )
            # We keep track of the denominator to compute the log-likelihood in the end
            alpha_log_denom += denom.log()

        # 3) Run backward phase
        beta[:, sequence_length-1] = 1
        for t in range(sequence_length - 1, 0, -1):
            beta[:, t-1] = normalize(
                (emission_probs[:, t] * beta[:, t]) @ self.markov.transition_probs.t()
            )

        # 4) Compute log-likelihood
        log_likeli = (alpha[:, sequence_length-1].log() + alpha_log_denom).logsumexp(-1).sum()

        return alpha, beta, -log_likeli

    def _packed_forward_backward(self, batch_sizes, emission_probs):
        M = self.num_states
        device = emission_probs.device
        max_length = batch_sizes.size(0)

        # 1) Initialize (empty) parameters
        alpha = torch.empty(batch_sizes.sum(), M, device=device)
        beta = torch.empty_like(alpha)
        alpha_log_denom = torch.zeros(batch_sizes[0], 1, device=device)

        # 2) Run forward phase
        data_offset = batch_sizes[0].item()
        alpha[:data_offset] = normalize(self.markov.initial_probs * emission_probs[:data_offset])

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

    ############################
    ### BAUM-WELCH ALGORITHM ###
    ############################
    def _estimate(self, sequences):
        N = sequences.size(0)
        S = sequences.size(1)

        # 1) Get alpha and beta
        emission_probs = self.emission.evaluate(sequences) # [N, S, K]
        alpha, beta, nll = self._forward_backward(N, S, emission_probs) # [N, S, K]

        # 2) Compute gamma, i.e. the smoothed probabilities for hidden states
        gamma = normalize(alpha * beta) # [N, S, K]

        # 3) Compute xi, i.e. transition probabilities between timesteps
        xi = self._compute_xi(alpha[:, :-1], beta[:, 1:], emission_probs[:, 1:])

        return gamma, xi, nll

    def _packed_estimate(self, sequences):
        num_sequences = sequences.batch_sizes[0].item()

        # 1) Get alpha and beta
        emission_probs = self.emission.evaluate(sequences.data) # [V, K]
        alpha, beta, nll = self._packed_forward_backward(sequences.batch_sizes, emission_probs)

        # 2) Compute gamma
        gamma = normalize(alpha * beta) # [V, K]

        # 3) Compute xi
        # 3.1) Drop last timestep of alpha
        alpha_ = packed_drop_last(alpha, sequences.batch_sizes)

        # 3.2) Drop first timestep of beta and emission probabilities
        beta_ = beta[num_sequences:]
        emission_probs_ = emission_probs[num_sequences:]

        # 3.3) Compute transitions
        xi = self._compute_xi(alpha_, beta_, emission_probs_)

        return gamma, xi, nll

    def _compute_xi(self, alpha_, beta_, emission_probs_):
        K = self.num_states
        alpha_ = alpha_.reshape(-1, K, 1)
        beta_ = (beta_ * emission_probs_).view(-1, 1, K)
        xi_num = torch.bmm(alpha_, beta_) * self.markov.transition_probs
        xi_num = xi_num.view(-1, K, K)
        return normalize(xi_num, [-1, -2])

    def _baum_welch_full(self, sequences):
        gamma, xi, nll = self._estimate(sequences)

        initial_probs = gamma[:, 0].mean(0)
        transition_probs = xi.sum(0) / gamma[:, :-1].sum([0, 1]).view(-1, 1)

        sequences_ = sequences.view(-1, *sequences.size()[2:])
        gamma_ = gamma.view(-1, *gamma.size()[2:])
        update = self.emission.maximize(sequences_, gamma_)

        return {
            'initial_probs': initial_probs,
            'transition_probs': transition_probs,
            'output_update': update,
            'neg_log_likelihood': nll / (sequences.size(0) * sequences.size(1))
        }

    def _packed_baum_welch_full(self, sequences):
        num_sequences = sequences.batch_sizes[0].item()

        gamma, xi, nll = self._packed_estimate(sequences)

        initial_probs = gamma[:num_sequences].mean(0)
        gamma_ = packed_drop_last(gamma, sequences.batch_sizes)
        transition_probs = xi.sum(0) / gamma_.sum(0).view(-1, 1)

        update = self.emission.maximize(sequences.data, gamma)

        return {
            'initial_probs': initial_probs,
            'transition_probs': transition_probs,
            'output_update': update,
            'neg_log_likelihood': nll / sequences.data.size(0)
        }

    def _baum_welch_batch(self, sequences, batch_size, weights):
        return 0

        # # 2) Maximize
        # # 2.1) Initial probabilities - simply average gamma values for first time step
        # initial_probs = gamma[:, 0].mean(0) # [K]

        # # 2.2) Transition probabilities
        # transition_probs_num = xi.sum([0, 1]) # [K, K]
        # transition_probs_denom = gamma.sum([0, 1]) # [K, K]

        # # 2.3) Output probabilities
        # update = self.emission.maximize(sequences, gamma)

        # return {
        #     'initial_probs': initial_probs,
        #     'transition_probs_num': transition_probs_num,
        #     'transition_probs_denom': transition_probs_denom,
        #     'neg_log_likelihood': nll,
        #     'output_update': update
        # }
