from dataclasses import dataclass
from typing import Tuple
import torch
from lightkit.nn import Configurable
from torch import jit, nn


@dataclass
class KMeansModelConfig:
    """
    Configuration class for a K-Means model.

    See also:
        :class:`KMeansModel`
    """

    #: The number of clusters.
    num_clusters: int
    #: The number of features of each cluster.
    num_features: int


class KMeansModel(Configurable[KMeansModelConfig], nn.Module):
    """
    PyTorch module for the K-Means model. The centroids managed by this model are non-trainable
    parameters.
    """

    def __init__(self, config: KMeansModelConfig):
        """
        Args:
            config: The configuration to use for initializing the module's buffers.
        """
        super().__init__(config)

        #: The centers of all clusters, buffer of shape ``[num_clusters, num_features].``
        self.centroids: torch.Tensor
        self.register_buffer("centroids", torch.empty(config.num_clusters, config.num_features))

        self.reset_parameters()

    @jit.unused
    def reset_parameters(self) -> None:
        """
        Resets the parameters of the KMeans model. It samples all cluster centers from a standard
        Normal.
        """
        nn.init.normal_(self.centroids)

    def forward(self, data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the distance of each datapoint to each centroid as well as the "inertia", the
        squared distance of each datapoint to its closest centroid.

        Args:
            data: A tensor of shape ``[num_datapoints, num_features]`` for which to compute the
                distances and inertia.

        Returns:
            - A tensor of shape ``[num_datapoints, num_centroids]`` with the distance from each
              datapoint to each centroid.
            - A tensor of shape ``[num_datapoints]`` with the assignments, i.e. the indices of
              each datapoint's closest centroid.
            - A tensor of shape ``[num_datapoints]`` with the inertia (squared distance to the
              closest centroid) of each datapoint.
        """
        distances = torch.cdist(data, self.centroids)
        assignments = distances.min(1, keepdim=True).indices
        inertias = distances.gather(1, assignments).square()
        return distances, assignments.squeeze(1), inertias.squeeze(1)
