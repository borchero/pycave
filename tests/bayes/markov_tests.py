import unittest
import torch
from pycave.bayes import MarkovModel

class TestMarkov(unittest.TestCase):
    """
    Test case to test the correctness of PyCave's Markov model.
    """

    def test_by_restore(self):
        """
        Test Markov model by restoring an existing one.
        """
        markov_reference = MarkovModel(num_states=4)
        markov_reference.initial_probs.set_(torch.as_tensor([
            0.5, 0.1, 0.2, 0.2
        ]))
        markov_reference.transition_probs.set_(torch.as_tensor([
            [0.3, 0.4, 0.2, 0.1],
            [0.6, 0.2, 0.1, 0.1],
            [0.1, 0.4, 0.4, 0.1],
            [0.2, 0.2, 0.3, 0.3]
        ]))

        torch.manual_seed(21)
        samples = markov_reference.sample(50_000, 20)

        markov = MarkovModel(num_states=4)
        markov.fit(samples)

        self.assertTrue(torch.allclose(
            markov_reference.initial_probs, markov.initial_probs,
            atol=0.01, rtol=0
        ))

        self.assertTrue(torch.allclose(
            markov_reference.transition_probs, markov.transition_probs,
            atol=0.01, rtol=0
        ))


if __name__ == '__main__':
    unittest.main()
