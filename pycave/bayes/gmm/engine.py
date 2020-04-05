import torch
import pyblaze.nn as xnn
from pycave.bayes._internal.utils import normalize

class GMMEngine(xnn.BaseEngine):
    """
    The GMMEngine can be used to train Gaussian mixture models on batches.
    """

    def __init__(self, model):
        super().__init__(model)

        self.cache = None
        self.best_nll = None
        self.requires_batching = None
        self.epoch = None

    def supports_multiple_gpus(self):
        return False

    def before_training(self, _, __):
        self.best_nll = float('inf')

    def before_epoch(self, current, num_iterations):
        self.requires_batching = num_iterations > 1
        self.epoch = current

        self.cache = {
            'count': 0,
            'component_weights': torch.zeros_like(self.model.component_weights),
            'gaussian': None,
            'neg_log_likelihood': 0
        }

    def after_epoch(self, _):
        # Update parameters
        self.model.component_weights.set_(self.cache['component_weights'])
        self.model.gaussian.apply(self.cache['gaussian'])

        # Read metadata
        eps = self.cache['eps']

        # Check for convergence
        nll = self.cache['neg_log_likelihood']
        if self.best_nll - nll < eps:
            raise xnn.CallbackException(f'Training converged after {self.epoch} iterations.')
        self.best_nll = nll

    def after_training(self):
        self.cache = None

    def train_batch(self, data, eps=0.01):
        # E-step: compute responsibilities
        responsibilities, nll = self.model(data)
        nll_ = nll.item() / data.size(0)

        # M-step: maximize
        gaussian_max = self.model.gaussian.maximize(data, responsibilities, self.requires_batching)
        component_weights = normalize(gaussian_max['state_sums'])

        # Store in cache
        new_count = self.cache['count'] + data.size(0)
        prev_weight = self.cache['count'] / new_count
        cur_weight = data.size(0) / new_count

        self.cache['count'] = new_count
        self.cache['gaussian'] = self.model.gaussian.update(gaussian_max, self.cache['gaussian'])
        self.cache['component_weights'] = \
            self.cache['component_weights'] * prev_weight + component_weights * cur_weight
        self.cache['neg_log_likelihood'] = \
            self.cache['neg_log_likelihood'] * prev_weight + (nll_ * cur_weight)

        # Attach metadata
        self.cache['eps'] = eps

    def eval_batch(self, data):
        return {
            'nll': self.model(data)[1].item(), # nll
            'n': data.data.size(0)
        }, None

    def predict_batch(self, data):
        # Get responsibilities and normalize them to get a distribution over components
        return normalize(self.model(data)[0])

    def collate_losses(self, _):
        nll = self.cache['neg_log_likelihood']
        return {'neg_log_likelihood': nll}

    def collate_predictions(self, predictions):
        sample = predictions[0]

        if isinstance(sample, dict):
            # Only negative log-likelihood
            nll_sum = sum([p['nll'] for p in predictions])
            n = sum([p['n'] for p in predictions])
            return {'neg_log_likelihood': nll_sum / n}

        # In this case, we return the responsibilities
        return torch.cat(predictions)

    def collate_targets(self, _):
        return None
