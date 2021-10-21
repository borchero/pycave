from __future__ import annotations
from typing import Any, Callable, cast, Dict, List, Optional
import numpy as np
import torch
from pycave.core.estimator import Estimator
from pycave.data import collate_sequences, collate_sequences_same_length, SequenceData
from .lightning_module import MarkovChainLightningModule
from .model import MarkovChainModel, MarkovChainModelConfig


class MarkovChain(Estimator[MarkovChainModel]):
    """
    Probabilistic model for observed state transitions.

    A Markov chain can be used to learn the initial probabilities of a set of states and the
    transition probabilities between them. It is similar to a hidden Markov model, only that the
    hidden states are known. More information is available
    `here <https://en.wikipedia.org/wiki/Markov_chain>`_.

    See also:
        .. currentmodule:: pycave.bayes.markov_chain
        .. autosummary::
            :nosignatures:
            :template: classes/pytorch_module.rst

            MarkovChainModel
            MarkovChainModelConfig
    """

    def __init__(
        self,
        num_states: Optional[int] = None,
        *,
        symmetric: bool = False,
        batch_size: Optional[int] = None,
        num_workers: int = 0,
        trainer_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            num_states: The number of states that the Markov chain has. If not provided, it will
                be derived automatically when calling :meth:`fit`. Note that this requires a pass
                through the data. Consider setting this option explicitly if you're fitting a lot
                of data.
            symmetric: Whether the transitions between states should be considered symmetric.
            batch_size: The batch size to use when fitting the model. If not provided, all data
                will be used as a single batch. You should consider setting this option if your
                data does not fit into memory.
            num_workers: The number of workers to use for loading the data. By default, it loads
                data on the main process.
            trainer_params: Initialization parameters to use when initializing a PyTorch Lightning
                trainer. This estimator sets an overridable default of `checkpoint_callback=False`
                and enforces `max_epochs=1`.
        """
        super().__init__(
            user_params=trainer_params,
            overwrite_params=dict(max_epochs=1),
        )

        self.num_states = num_states
        self.symmetric = symmetric
        self.batch_size = batch_size
        self.num_workers = num_workers

    def fit(self, sequences: SequenceData) -> MarkovChain:
        """
        Fits the Markov chain on the provided data and returns the fitted estimator.

        Args:
            sequences: The sequences to fit the Markov chain on.

        Returns:
            The fitted Markov chain.
        """
        config = MarkovChainModelConfig(
            num_states=self.num_states or _get_num_states(sequences),
        )
        self._model = MarkovChainModel(config)

        self._trainer().fit(
            MarkovChainLightningModule(self.model_, self.symmetric),
            self._init_data_loader(sequences, for_training=True),
        )
        return self

    def sample(self, num_sequences: int, sequence_length: int) -> torch.Tensor:
        """
        Samples state sequences from the fitted Markov chain.

        Args:
            num_sequences: The number of sequences to sample.
            sequence_length: The length of the sequences to sample.

        Returns:
            The sampled sequences as a tensor of shape ``[num_sequences, sequence_length]``.
        """
        return self.model_.sample(num_sequences, sequence_length)

    def score(self, sequences: SequenceData) -> float:
        """
        Computes the average negative log-likelihood (NLL) of observing the provided sequences. If
        you want to have NLLs for each individual sequence, use :meth:`score_samples` instead.

        Args:
            sequences: The sequences for which to compute the average log-probability.

        Returns:
            The average NLL for all sequences.
        """
        result = self._trainer().test(
            MarkovChainLightningModule(self.model_),
            self._init_data_loader(sequences, for_training=False),
            verbose=False,
        )
        return result[0]["nll"]

    def score_samples(self, sequences: SequenceData) -> torch.Tensor:
        """
        Computes the average negative log-likelihood (NLL) of observing the provided sequences.

        Args:
            sequences: The sequences for which to compute the NLL.

        Returns:
            A tensor of shape ``[num_sequences]`` with the NLLs for each individual sequence.
        """
        result = self._trainer().predict(
            MarkovChainLightningModule(self.model_),
            self._init_data_loader(sequences, for_training=False),
            return_predictions=True,
        )
        return torch.stack(cast(List[torch.Tensor], result))

    def _data_collate_fn(self, for_tensor: bool) -> Optional[Callable[[Any], Any]]:
        if for_tensor:
            return collate_sequences_same_length
        return collate_sequences


def _get_num_states(data: SequenceData) -> int:
    if isinstance(data, np.ndarray):
        assert data.dtype == np.int64, "array states must have type `np.int64`"
        return int(data.max() + 1)
    if isinstance(data, torch.Tensor):
        assert data.dtype == torch.long, "tensor states must have type `torch.long`"
        return int(data.max().item() + 1)
    return max(_get_num_states(entry) for entry in data)
