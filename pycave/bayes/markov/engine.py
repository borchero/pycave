import torch
import pyblaze.nn as xnn
from pycave.bayes._internal.utils import normalize

class MarkovModelEngine(xnn.Engine):
    """
    This engine can be used to train a Markov model on batches of data.
    """

    def __init__(self, model):
        super().__init__(model)

        self.cache = None

    def supports_multiple_gpus(self):
        return False

    def before_epoch(self, _, __):
        # Initialize cache
        self.cache = {
            'initial_counts': torch.zeros_like(self.model.initial_probs),
            'transition_counts': torch.zeros_like(self.model.transition_probs).view(-1)
        }

    def after_epoch(self, _):
        # This function sets the model's parameters and immediately terminates training.

        # Initial probs
        initial_probs = normalize(self.cache['initial_counts'])
        self.model.initial_probs.set_(initial_probs)

        # Transition probs
        tr_size = self.model.transition_probs.size()
        transition_counts = self.cache['transition_counts'].view(*tr_size)

        # Symmetric
        if self.cache['symmetric']:
            transition_counts += transition_counts.t()
        transition_probs = normalize(transition_counts)

        # Teleportation
        if self.cache['teleport_alpha'] > 0:
            teleport_factor = self.cache['teleport_alpha'] / tr_matrix.size(0)
            teleport_matrix = torch.ones_like(tr_matrix)
            beta = 1 - self.cache['teleport_alpha']
            tr_matrix = (tr_matrix - teleport_factor * teleport_matrix) / beta

        self.model.transition_probs.set_(transition_probs)

        # Terminate training
        raise xnn.CallbackException('MarkovModel training does not need iterations')

    def after_training(self):
        # Clear the cache
        self.cache = None

    def train_batch(self, data, symmetric=False, teleport_alpha=0):
        # Accumulates statistics about the data given as packed sequence.
        num_sequences = data.batch_sizes[0].item()
        num_states = self.model.num_states

        # Initial state probabilities
        initial_states = data.data[:num_sequences]
        initial_counts = torch.bincount(initial_states, minlength=self.model.num_states)
        self.cache['initial_counts'] += initial_counts.float()

        # Transition probabilities
        offset = num_sequences
        for prev_size, size in zip(data.batch_sizes, data.batch_sizes[1:]):
            sources = data.data[offset-prev_size: offset-prev_size+size]
            targets = data.data[offset: offset+size]
            transitions = sources * num_states + targets
            values = torch.ones_like(transitions, dtype=torch.float)
            self.cache['transition_counts'].scatter_add_(0, transitions, values)

        # Metadata
        self.cache['symmetric'] = symmetric
        self.cache['teleport_alpha'] = teleport_alpha

        # Return nothing (no losses reported)

    def eval_batch(self, data):
        return {
            'nll': self.model(data).item(),
            'n': data.data.size(0)
        }

    def predict_batch(self, data):
        raise ValueError("Markov models do not support predictions")

    def collate_losses(self, losses):
        return {}

    def collate_predictions(self, predictions):
        nll_sum = sum([p['nll'] for p in predictions])
        n = sum([p['n'] for p in predictions])
        return {'neg_log_likelihood': nll_sum / n}
