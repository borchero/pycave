# pylint: disable=missing-function-docstring
import torch
from torch import jit
from torchgmm.clustering.kmeans import KMeansModel, KMeansModelConfig


def test_compile():
    config = KMeansModelConfig(num_clusters=2, num_features=5)
    model = KMeansModel(config)
    jit.script(model)


def test_forward():
    config = KMeansModelConfig(num_clusters=2, num_features=2)
    model = KMeansModel(config)
    model.centroids.copy_(torch.as_tensor([[0.0, 0.0], [2.0, 2.0]]))

    X = torch.as_tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [-1.0, 4.0]])
    distances, assignments, inertias = model.forward(X)

    expected_distances = torch.as_tensor([[0.0, 8.0], [2.0, 2.0], [8.0, 0.0], [17.0, 13.0]]).sqrt()
    expected_assignments = torch.as_tensor([0, 0, 1, 1])
    expected_inertias = torch.as_tensor([0.0, 2.0, 0.0, 13.0])

    assert torch.allclose(distances, expected_distances)
    assert torch.all(assignments == expected_assignments)
    assert torch.allclose(inertias, expected_inertias)
