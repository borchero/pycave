import torch
from pycave.clustering import KMeans


def test_recover():
    torch.manual_seed(42)

    estimator = KMeans(2)
    data = torch.cat([torch.randn(1000, 4) * 0.1 - 1, torch.randn(1000, 4) * 0.1 + 1])
    estimator.fit(data)

    ordering = estimator.model_.centroids[:, 0].argsort()
    expected = torch.as_tensor([[-1.0, -1.0, -1.0, -1.0], [1.0, 1.0, 1.0, 1.0]])[ordering]

    assert torch.allclose(estimator.model_.centroids, expected, atol=1e-2)
