from __future__ import annotations
from typing import Any, cast, Dict, List, Optional
import torch
from pycave.core import Estimator
from pycave.data import TabularData
from .ligthning_module import KMeansLightningModule
from .model import KMeansModel, KMeansModelConfig
from .types import KMeansInitStrategy


class KMeans(Estimator[KMeansModel]):
    """
    Model for clustering data into a predefined number of clusters.

    See also:
        .. currentmodule:: pycave.clustering.kmeans
        .. autosummary::
            :nosignatures:
            :template: classes/pytorch_module.rst

            KMeansModel
            KMeansModelConfig
    """

    #: A boolean indicating whether the model converged during training.
    converged_: bool
    #: The number of iterations the model was fitted for, excluding initialization.
    num_iter_: int
    #: The mean squared distance of all datapoints to their closest cluster centers.
    inertia_: float

    def __init__(
        self,
        num_clusters: int = 1,
        *,
        init_strategy: KMeansInitStrategy = "kmeans++",
        convergence_tolerance: float = 1e-3,
        batch_size: Optional[int] = None,
        num_workers: int = 0,
        trainer_params: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            num_clusters: The number of clusters.
            init_strategy: The strategy for initializing centroids.
            convergence_tolerance: Training is conducted until the decrease in change of per-
                datapoint inertia falls below this value.
            batch_size: The batch size to use when fitting the model. If not provided, the full
                data will be used as a single batch. Set this if the full data does not fit into
                memory.
            num_workers: The number of workers to use for loading the data. Only used if a PyTorch
                dataset is passed to :meth:`fit` or related methods.
            trainer_params: Initialization parameters to use when initializing a PyTorch Lightning
                trainer. This estimator sets the following overridable defaults:

                - ``max_epochs=300``
                - ``checkpoint_callback=False``
                - ``log_every_n_steps=1``
        """
        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            default_params=dict(
                max_epochs=300,
                checkpoint_callback=False,
                log_every_n_steps=1,
            ),
            user_params=trainer_params,
        )

        # We need to account for the initialization epochs in `max_epochs`
        self._trainerparams = {
            k: (
                v + (1 if init_strategy == "random" else num_clusters)
                if k in ("min_epochs", "max_epochs")
                else v
            )
            for k, v in self._trainerparams.items()
        }

        # Assign other properties
        self.num_clusters = num_clusters
        self.init_strategy = init_strategy
        self.convergence_tolerance = convergence_tolerance

    def fit(self, data: TabularData) -> KMeans:
        """
        Fits the KMeans model on the provided data by running the EM algorithm.

        Args:
            data: The tabular data to fit on. The dimensionality of the KMeans model is
                automatically inferred from this data.

        Returns:
            The fitted KMeans model.
        """
        self._init_trainer()

        # Initialize model
        num_features = len(data[0])
        config = KMeansModelConfig(
            num_clusters=self.num_clusters,
            num_features=num_features,
        )
        self._model = KMeansModel(config)

        # Fit model
        loader = self._init_data_loader(data, for_training=True)
        module = KMeansLightningModule(
            self.model_,
            init_strategy=self.init_strategy,  # type: ignore
            tol=self.convergence_tolerance,
            batch_training=self._uses_batch_training(loader),
        )
        self._trainer.fit(module, loader)

        # Assign convergence properties
        self.num_iter_ = (
            module.current_epoch
            + 1
            - (self.num_clusters if self.init_strategy == "kmeans++" else 1)
        )
        self.converged_ = module.current_epoch + 1 < cast(int, self._trainer.max_epochs)
        self.inertia_ = cast(float, self._trainer.callback_metrics["inertia"].item())
        return self

    def predict(self, data: TabularData) -> torch.Tensor:
        """
        Predicts the closest cluster for each item provided.

        Args:
            data: The datapoints for which to predict the clusters.

        Returns:
            Tensor of shape ``[num_datapoints]`` with the index of the closest cluster for each
            datapoint.

        Note:
            When calling this function in a multi-process environment, each process receives only
            a subset of the predictions. If you want to aggregate predictions, make sure to gather
            the values returned from this method. Take care that this only works when the number
            of predictions in each process is equal, i.e. if the provided data is divisible by the
            number of processes.
        """
        result = self._trainer.predict(
            KMeansLightningModule(self.model_),
            self._init_data_loader(data, for_training=False),
        )
        return torch.cat(cast(List[torch.Tensor], result))

    def score(self, data: TabularData) -> float:
        """
        Computes the average inertia of all the provided datapoints. That is, it computes the mean
        squared distance to each datapoint's closest centroid.

        Args:
            data: The data for which to compute the average inertia.

        Returns:
            The average inertia.
        """
        result = self._trainer.test(
            KMeansLightningModule(self.model_),
            self._init_data_loader(data, for_training=False),
        )
        return result[0]["inertia"]
