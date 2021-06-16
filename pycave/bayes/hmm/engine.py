import torch
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence
import pyblaze.nn as xnn
from pyblaze.utils.stdlib import flatten
from pycave.bayes._internal.utils import packed_drop_first, packed_drop_last, packed_get_first, \
    normalize

class HMMEngine(xnn.Engine):
    """
    The HMMEngine can be used to train a hidden Markov models on batches yielded by a data loader.
    """

    def __init__(self, model):
        super().__init__(model)

        self.requires_batching = None
        self.cache = None
        self.best_nll = None
        self.patience = 0
        self.epoch = None

    def supports_multiple_gpus(self):
        return False

    def before_training(self, _, __):
        # Set up early stopping criterion
        self.best_nll = float('inf')
        self.patience = 0

    def before_epoch(self, current, num_iterations):
        self.epoch = current

        # Set up result cache to update parameters
        if num_iterations == 1:
            # Essentially, there is no batching
            self.requires_batching = False
            self.cache = None
        else:
            # Otherwise, we need to initialize "containers" to accumulate intermediate results
            self.requires_batching = True

            device = self.model.markov.transition_probs.device
            num_states = self.model.num_states
            self.cache = {
                'num_sequences': 0,
                'num_datapoints': 0,
                'initial_probs': torch.zeros(num_states, device=device),
                'tr_probs_num': torch.zeros(num_states, num_states, device=device),
                'output_update': None,
                'nll_sum': 0
            }

    def after_epoch(self, _):
        # This function acts as final "maximize" step of the Baum-Welch algorithm if batching was
        # performed. Otherwise, it just updates the parameters of the model.
        if not self.requires_batching:
            self.model.markov.initial_probs.set_(self.cache['initial_probs'])
            self.model.markov.transition_probs.set_(self.cache['transition_probs'])
            self.model.emission.apply(self.cache['output_update'])
            nll = self.cache['neg_log_likelihood']
        else:
            self.model.markov.initial_probs.set_(self.cache['initial_probs'])
            self.model.markov.transition_probs.set_(normalize(self.cache['tr_probs_num']))
            self.model.emission.apply(self.cache['output_update'])
            nll = self.cache['nll_sum'] / self.cache['num_datapoints']

        # This metadata field is always present
        eps = self.cache['eps']
        patience = self.cache['patience']

        # Check for early stopping
        if self.best_nll - nll < eps:
            if self.patience < patience:
                self.patience += 1
            else:
                raise xnn.CallbackException(f'Training converged after {self.epoch} iterations.')
        else:
            self.best_nll = nll
            self.patience = 0

    def after_training(self):
        # Clear the cache
        self.cache = None

    def train_batch(self, data, eps=0.01, patience=0):
        # This function acts as "expect" step of the Baum-Welch algorithm as well as computing
        # intermediate results for the M-step. The data is expected to be a packed sequence on the
        # correct device.

        # 1) Expect step
        emission_probs, alpha, beta, nll = self.model(data, smooth=True, return_emission=True)
        gamma = normalize(alpha * beta)

        alpha_ = packed_drop_last(alpha, data.batch_sizes)
        beta_ = packed_drop_first(beta, data.batch_sizes)
        emission_probs_ = packed_drop_first(emission_probs, data.batch_sizes)
        xi = self._compute_xi(alpha_, beta_, emission_probs_)

        # 2) Maximize step
        initial_probs = packed_get_first(gamma, data.batch_sizes).mean(0)
        transition_probs_num = xi.sum(0)
        transition_probs = normalize(transition_probs_num)
        output_update = self.model.emission.maximize(data.data, gamma, self.requires_batching)
        nll = nll.item()

        # 3) Update cache depending on single-batch or multi-batch setting
        if not self.requires_batching:
            # We know that we only get a single batch
            self.cache = {
                'initial_probs': initial_probs,
                'transition_probs': transition_probs,
                'output_update': output_update,
                'neg_log_likelihood': nll / data.data.size(0)
            }
        else:
            num_seqs = data.batch_sizes[0].item()
            prv_wgt, cur_wgt = self._weights_for_updated_count('num_sequences', num_seqs)

            self.cache['initial_probs'] = \
                self.cache['initial_probs'] * prv_wgt + initial_probs * cur_wgt
            self.cache['tr_probs_num'] += transition_probs_num
            self.cache['output_update'] = \
                self.model.emission.update(output_update, self.cache['output_update'])
            self.cache['nll_sum'] += nll
            self.cache['num_datapoints'] += data.data.size(0)

        # 4) Add metadata to cache (this is a no-op for all but the first batch)
        self.cache['eps'] = eps
        self.cache['patience'] = patience

        # We return nothing as we will set the loss ourselves in `collate_losses`

    def eval_batch(self, data):
        return {
            'nll': self.model(data)[1].item(), # just get the nll
            'n': data.data.size(0)
        }

    def predict_batch(self, data, smooth=False):
        return {
            'out': self.model(data, smooth=smooth),
            'idx': data.unsorted_indices,
            'bs': data.batch_sizes
        }

    def collate_losses(self, _):
        if not self.requires_batching:
            value = self.cache['neg_log_likelihood']
        else:
            value = self.cache['nll_sum'] / self.cache['num_datapoints']

        return {'neg_log_likelihood': value}

    def collate_predictions(self, predictions):
        sample = predictions[0]

        if 'nll' in sample:
            # Only negative log-likelihood, divide by the number of datapoints
            nll_sum = sum([p['nll'] for p in predictions])
            n = sum([p['n'] for p in predictions])
            return {'neg_log_likelihood': nll_sum / n}

        if len(sample['out']) == 2:
            # Simple forward application, just concatenate the first values
            return torch.cat([
                p['out'][0][p['idx']] if p['idx'] is not None else p['out'][0]
                for p in predictions
            ])

        # Smoothed application, concatenate for alpha and beta and compute gamma
        return flatten([
            self._rearrange_prediction_sequence(p) for p in predictions
        ])

    def _rearrange_prediction_sequence(self, item):
        gamma = normalize(item['out'][0] * item['out'][1])
        packed = PackedSequence(data=gamma, batch_sizes=item['bs'])
        padded, lengths = pad_packed_sequence(packed, batch_first=True)
        if item['idx'] is not None:
            return [padded[i, :lengths[i]] for i in item['idx']]
        return [padded[i, :lengths[i]] for i in range(lengths.size(0))]

    def _compute_xi(self, alpha_, beta_, emission_probs_):
        K = self.model.num_states
        alpha_ = alpha_.reshape(-1, K, 1)
        beta_ = (beta_ * emission_probs_).view(-1, 1, K)
        xi_num = torch.bmm(alpha_, beta_) * self.model.markov.transition_probs
        xi_num = xi_num.view(-1, K, K)
        return normalize(xi_num, [-1, -2])

    def _weights_for_updated_count(self, key, num):
        prev_count = self.cache[key]
        current_count = prev_count + num

        prev_weight = prev_count / current_count
        current_weight = num / current_count

        self.cache[key] = current_count

        return prev_weight, current_weight
