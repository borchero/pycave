import torch
from pycave.bayes import GaussianMixture


def test_fit_automatic_config():
    gm = GaussianMixture(covariance_type="spherical")
    data = torch.randn(1000, 4)
    gm.fit(data)
    assert gm.model_.config.num_components == 1
    assert gm.model_.config.covariance_type == "spherical"
    assert gm.model_.config.num_features == 4


# def test_fit():
#     data = torch.randn(1000, 4)
#     gm = GaussianMixture(num_components=3, covariance_type="spherical")
#     gm.fit(data)
