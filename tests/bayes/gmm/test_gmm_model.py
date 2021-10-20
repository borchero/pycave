# pylint: disable=missing-function-docstring
from torch import jit
from pycave.bayes.gmm import GaussianMixtureModel, GaussianMixtureModelConfig


def test_compile():
    config = GaussianMixtureModelConfig(num_components=2, num_features=3, covariance_type="full")
    model = GaussianMixtureModel(config)
    jit.script(model)
