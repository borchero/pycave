# pylint: disable=missing-function-docstring
import math
import torch
from torch import jit
from torch.nn.utils.rnn import pack_padded_sequence
from pycave.bayes.markov_chain import MarkovChainModel, MarkovChainModelConfig


def test_compile():
    config = MarkovChainModelConfig(num_states=2)
    model = MarkovChainModel(config)
    jit.script(model)


def test_forward_tensor():
    model = _get_default_model()
    sequences = torch.as_tensor([[1, 0, 0, 1], [0, 1, 1, 1]])
    expected = torch.as_tensor(
        [
            math.log(0.8) + math.log(0.1) + math.log(0.5) + math.log(0.5),
            math.log(0.2) + math.log(0.5) + math.log(0.9) + math.log(0.9),
        ]
    )
    assert torch.allclose(expected, model(sequences))


def test_forward_packed_sequence():
    model = _get_default_model()
    sequences = torch.as_tensor([[1, 0, 0, 1], [0, 1, 1, -1]])
    packed_sequences = pack_padded_sequence(sequences.t(), torch.Tensor([4, 3]))
    expected = torch.as_tensor(
        [
            math.log(0.8) + math.log(0.1) + math.log(0.5) + math.log(0.5),
            math.log(0.2) + math.log(0.5) + math.log(0.9),
        ]
    )
    assert torch.allclose(expected, model(packed_sequences))


def test_sample():
    torch.manual_seed(42)
    model = _get_default_model()
    n = 100000
    samples = model.sample(n, 3)
    assert math.isclose((samples[:, 0] == 0).sum() / n, 0.2, abs_tol=0.01)


def test_stationary_distribution():
    model = _get_default_model()
    tol = 1e-7
    sd = model.stationary_distribution(tol=tol)
    assert math.isclose(sd[0].item(), 1 / 6, abs_tol=tol)
    assert math.isclose(sd[1].item(), 5 / 6, abs_tol=tol)


# -------------------------------------------------------------------------------------------------


def _get_default_model() -> MarkovChainModel:
    config = MarkovChainModelConfig(num_states=2)
    model = MarkovChainModel(config)

    model.initial_probs.copy_(torch.as_tensor([0.2, 0.8]))
    model.transition_probs.copy_(torch.as_tensor([[0.5, 0.5], [0.1, 0.9]]))
    return model
