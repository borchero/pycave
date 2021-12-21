from __future__ import annotations
import logging
from typing import Any, cast, Dict, List, Optional
import numpy as np
import torch
from lightkit import BaseEstimator
from lightkit.data import DataLoader, dataset_from_tensors
from torch.nn.utils.rnn import PackedSequence
from torch.utils.data import Dataset
from .lightning_module import MarkovChainLightningModule
from .model import MarkovChainModel, MarkovChainModelConfig
from .types import collate_sequences, collate_sequences_same_length, SequenceData

logger = logging.getLogger(__name__)


class MarkovChain(BaseEstimator[MarkovChainModel]):
    """
    Probabilistic model for observed state transitions. The Markov chain is similar to the hidden
    Markov model, only that the hidden states are known. More information on the Markov chain is
    available on `Wikipedia <https://en.wikipedia.org/wiki/Markov_chain>`_.

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
        trainer_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            num_states: The number of states that the Markov chain has. If not provided, it will
                be derived automatically when calling :meth:`fit`. Note that this requires a pass
                through the data. Consider setting this option explicitly if you're fitting a lot
                of data.
            symmetric: Whether the transitions between states should be considered symmetric.
            batch_size: The batch size to use when fitting the model. If not provided, the full
                data will be used as a single batch. Set this if the full data does not fit into
                memory.
            num_workers: The number of workers to use for loading the data. Only used if a PyTorch
                dataset is passed to :meth:`fit` or related methods.
            trainer_params: Initialization parameters to use when initializing a PyTorch Lightning
                trainer. By default, it disables various stdout logs unless PyCave is configured to
                do verbose logging. Checkpointing and logging are disabled regardless of the log
                level. This estimator further enforces the following parameters:

                - ``max_epochs=1``
        """
        super().__init__(
            user_params=trainer_params,
            overwrite_params=dict(max_epochs=1),
        )

        self.num_states = num_states
        self.symmetric = symmetric
        self.batch_size = batch_size

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

        logger.info("Fitting Markov chain...")
        self.trainer().fit(
            MarkovChainLightningModule(self.model_, self.symmetric),
            self._init_data_loader(sequences),
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

        Note:
            This method does not parallelize across multiple processes, i.e. performs no
            synchronization.
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

        Note:
            See :meth:`score_samples` to obtain the NLL values for individual sequences.
        """
        result = self.trainer().test(
            MarkovChainLightningModule(self.model_),
            self._init_data_loader(sequences),
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

        Attention:
            When calling this function in a multi-process environment, each process receives only
            a subset of the predictions. If you want to aggregate predictions, make sure to gather
            the values returned from this method.
        """
        result = self.trainer().predict(
            MarkovChainLightningModule(self.model_),
            self._init_data_loader(sequences),
            return_predictions=True,
        )
        return torch.stack(cast(List[torch.Tensor], result))

    def _init_data_loader(self, sequences: SequenceData) -> DataLoader[PackedSequence]:
        if isinstance(sequences, Dataset):
            return DataLoader(
                sequences,
                batch_size=self.batch_size or len(sequences),  # type: ignore
                collate_fn=collate_sequences,  # type: ignore
            )

        return DataLoader(  # type: ignore
            dataset_from_tensors(sequences),
            batch_size=self.batch_size or len(sequences),
            collate_fn=collate_sequences_same_length,
        )


def _get_num_states(data: SequenceData) -> int:
    if isinstance(data, np.ndarray):
        assert data.dtype == np.int64, "array states must have type `np.int64`"
        return int(data.max() + 1)
    if isinstance(data, torch.Tensor):
        assert data.dtype == torch.long, "tensor states must have type `torch.long`"
        return int(data.max().item() + 1)
    return max(_get_num_states(entry) for entry in data)
