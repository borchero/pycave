import torch
from torch.nn.utils.rnn import PackedSequence
from torchmetrics import AverageMeter
from pycave.bayes.markov_chain.metrics import StateCountAggregator
from pycave.utils import NonparametricLightningModule
from .model import MarkovChainModel


class MarkovChainLightningModule(NonparametricLightningModule):
    """
    Lightning module for training and evaluating a Markov chain.
    """

    def __init__(self, model: MarkovChainModel, symmetric: bool = False):
        """
        Args:
            model: The model to train or evaluate.
            symmetric: Whether transition probabilities should be symmetric.
        """
        super().__init__()

        self.model = model
        self.symmetric = symmetric

        self.aggregator = StateCountAggregator(
            num_states=self.model.config.num_states,
            symmetric=self.symmetric,
            dist_sync_fn=self.all_gather,
        )
        self.metric_nll = AverageMeter(dist_sync_fn=self.all_gather)

    def on_train_epoch_start(self) -> None:
        self.aggregator.reset()

    def nonparametric_training_step(self, batch: PackedSequence, _batch_idx: int) -> None:
        self.aggregator.update(batch)

    def nonparametric_training_epoch_end(self) -> None:
        initial_probs, transition_probs = self.aggregator.compute()
        self.model.initial_probs.copy_(initial_probs)
        self.model.transition_probs.copy_(transition_probs)

    def test_step(self, batch: PackedSequence, _batch_idx: int) -> None:
        log_probs = self.model.forward(batch)
        self.metric_nll.update(-log_probs)
        self.log("nll", self.metric_nll)

    def predict_step(  # pylint: disable=signature-differs
        self, batch: PackedSequence, batch_idx: int
    ) -> torch.Tensor:
        return -self.model(batch)
