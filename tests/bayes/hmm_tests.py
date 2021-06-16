import unittest
import torch
from pycave.bayes import HMM

class TestHMM(unittest.TestCase):
    """
    Test case to test the correctness of PyCave's HMM.
    """

    def test_forward(self):
        """
        Test Discrete HMM forward method.
        """
        hmm = HMM(num_states=2, output='discrete', output_num_states=3)
        hmm.markov.initial_probs.set_(torch.as_tensor([0.8, 0.2]))
        hmm.markov.transition_probs.set_(torch.as_tensor([[0.8, 0.2], [0.3, 0.7]]))
        hmm.emission.probabilities.set_(torch.as_tensor([[0.2, 0.7, 0.1], [0.6, 0.1, 0.3]]))

        # 1) Simple forward application
        seq = torch.as_tensor([[2, 2, 0], [1, 2, 1]])
        out = hmm.predict(seq)

        self.assertAlmostEqual(out[0, 0].item(), 0.2213, places=4)
        self.assertAlmostEqual(out[0, 1].item(), 0.7787, places=4)

        self.assertAlmostEqual(out[1, 0].item(), 0.9037, places=4)
        self.assertAlmostEqual(out[1, 1].item(), 0.0963, places=4)

        # 2) Packed forward application (sorted from largest to smallest sequence)
        seqs = torch.nn.utils.rnn.pack_sequence([
            torch.as_tensor([1, 2, 1, 1]), torch.as_tensor([2, 2, 0]), torch.as_tensor([0, 1])
        ])
        out = hmm.predict(seqs)

        self.assertAlmostEqual(out[0, 0].item(), 0.9550, places=4)
        self.assertAlmostEqual(out[0, 1].item(), 0.0450, places=4)

        self.assertAlmostEqual(out[1, 0].item(), 0.2213, places=4)
        self.assertAlmostEqual(out[1, 1].item(), 0.7787, places=4)

        self.assertAlmostEqual(out[2, 0].item(), 0.9082, places=4)
        self.assertAlmostEqual(out[2, 1].item(), 0.0918, places=4)

        # 3) Packed forward application (unsorted)
        seqs = torch.nn.utils.rnn.pack_sequence([
            torch.as_tensor([0, 1]), torch.as_tensor([1, 2, 1, 1]), torch.as_tensor([2, 2, 0])
        ], enforce_sorted=False)
        out = hmm.predict(seqs)

        self.assertAlmostEqual(out[0, 0].item(), 0.9082, places=4)
        self.assertAlmostEqual(out[0, 1].item(), 0.0918, places=4)

        self.assertAlmostEqual(out[1, 0].item(), 0.9550, places=4)
        self.assertAlmostEqual(out[1, 1].item(), 0.0450, places=4)

        self.assertAlmostEqual(out[2, 0].item(), 0.2213, places=4)
        self.assertAlmostEqual(out[2, 1].item(), 0.7787, places=4)

    def test_smoothing(self):
        """
        Test Discrete HMM smoothing (forward-backward algorithm).
        """
        hmm = HMM(num_states=2, output='discrete', output_num_states=3)
        hmm.markov.initial_probs.set_(torch.as_tensor([0.8, 0.2]))
        hmm.markov.transition_probs.set_(torch.as_tensor([[0.8, 0.2], [0.3, 0.7]]))
        hmm.emission.probabilities.set_(torch.as_tensor([[0.2, 0.7, 0.1], [0.6, 0.1, 0.3]]))

        # 1) Simple smoothing
        seq = torch.as_tensor([[2, 2, 0], [1, 2, 1]])
        out = hmm.predict(seq, smooth=True)

        self.assertTrue(torch.allclose(
            out[0], torch.as_tensor([[0.3847, 0.6153], [0.2156, 0.7844], [0.2213, 0.7787]]),
            rtol=0, atol=1e-4
        ))
        self.assertTrue(torch.allclose(
            out[1], torch.as_tensor([[0.9587, 0.0413], [0.7133, 0.2867], [0.9037, 0.0963]]),
            rtol=0, atol=1e-4
        ))

        # 2) Packed smoothing (sorted from largest to smallest sequence)
        seqs = torch.nn.utils.rnn.pack_sequence([
            torch.as_tensor([1, 2, 1, 1]), torch.as_tensor([2, 2, 0]), torch.as_tensor([0, 1])
        ])
        out = hmm.predict(seqs, smooth=True)

        self.assertTrue(torch.allclose(
            out[0], torch.as_tensor([
                [0.9611, 0.0389], [0.7373, 0.2627], [0.9511, 0.0489], [0.9550, 0.0450]
            ]), rtol=0, atol=1e-4
        ))
        self.assertTrue(torch.allclose(
            out[1], torch.as_tensor([[0.3847, 0.6153], [0.2156, 0.7844], [0.2213, 0.7787]]),
            rtol=0, atol=1e-4
        ))
        self.assertTrue(torch.allclose(
            out[2], torch.as_tensor([[0.7342, 0.2658], [0.9082, 0.0918]]),
            rtol=0, atol=1e-4
        ))

        # 3) Packed smoothing (unsorted)
        seqs = torch.nn.utils.rnn.pack_sequence([
            torch.as_tensor([0, 1]), torch.as_tensor([1, 2, 1, 1]), torch.as_tensor([2, 2, 0])
        ], enforce_sorted=False)
        out = hmm.predict(seqs, smooth=True)

        self.assertTrue(torch.allclose(
            out[0], torch.as_tensor([[0.7342, 0.2658], [0.9082, 0.0918]]),
            rtol=0, atol=1e-4
        ))
        self.assertTrue(torch.allclose(
            out[1], torch.as_tensor([
                [0.9611, 0.0389], [0.7373, 0.2627], [0.9511, 0.0489], [0.9550, 0.0450]
            ]), rtol=0, atol=1e-4
        ))
        self.assertTrue(torch.allclose(
            out[2], torch.as_tensor([[0.3847, 0.6153], [0.2156, 0.7844], [0.2213, 0.7787]]),
            rtol=0, atol=1e-4
        ))

    def test_by_restore_gaussian(self):
        """
        Test Gaussian HMM with spherical covariance by restoring an existing HMM.
        """
        config = {
            'num_states': 3,
            'output_dim': 2,
            'output_covariance': 'spherical'
        }

        hmm_reference = HMM(**config)

        hmm_reference.markov.initial_probs.set_(torch.as_tensor(
            [0.75, 0.25, 0], dtype=torch.float
        ))
        hmm_reference.markov.transition_probs.set_(torch.as_tensor([
            [0, 1, 0],
            [0.5, 0, 0.5],
            [1, 0, 0]
        ], dtype=torch.float))

        hmm_reference.emission.means.set_(torch.as_tensor([
            [-1, -1], [0, 1], [1, -1]
        ], dtype=torch.float))
        hmm_reference.emission.covars.set_(torch.as_tensor(
            [0.1, 0.25, 0.2]
        ))

        torch.manual_seed(42)
        sequences = hmm_reference.sample(8192, 8)

        hmm = HMM(**config)
        hmm.fit(sequences)

        order = hmm.markov.initial_probs.argsort(descending=True)

        self.assertTrue(torch.allclose(
            hmm.markov.initial_probs[order], hmm_reference.markov.initial_probs,
            atol=0.01, rtol=0
        ))

        self.assertTrue(torch.allclose(
            hmm.markov.transition_probs[order][:, order], hmm_reference.markov.transition_probs,
            atol=0.01, rtol=0
        ))

        self.assertTrue(torch.allclose(
            hmm.emission.means[order], hmm_reference.emission.means,
            atol=0.05, rtol=0
        ))

        self.assertTrue(torch.allclose(
            hmm.emission.covars[order], hmm_reference.emission.covars,
            atol=0.02, rtol=0
        ))

    def test_by_batch_restore_gaussian(self):
        """
        Test Gaussian HMM with spherical covariance by restoring an existing HMM via batch training.
        """
        config = {
            'num_states': 3,
            'output_dim': 2,
            'output_covariance': 'spherical'
        }

        hmm_reference = HMM(**config)

        hmm_reference.markov.initial_probs.set_(torch.as_tensor(
            [0.75, 0.25, 0], dtype=torch.float
        ))
        hmm_reference.markov.transition_probs.set_(torch.as_tensor([
            [0, 1, 0],
            [0.5, 0, 0.5],
            [1, 0, 0]
        ], dtype=torch.float))

        hmm_reference.emission.means.set_(torch.as_tensor([
            [-1, -1], [0, 1], [1, -1]
        ], dtype=torch.float))
        hmm_reference.emission.covars.set_(torch.as_tensor(
            [0.1, 0.25, 0.2]
        ))

        torch.manual_seed(42)
        sequences = hmm_reference.sample(8192, 8)

        hmm = HMM(**config)
        hmm.fit(sequences.chunk(32))

        order = hmm.markov.initial_probs.argsort(descending=True)

        self.assertTrue(torch.allclose(
            hmm.markov.initial_probs[order], hmm_reference.markov.initial_probs,
            atol=0.01, rtol=0
        ))

        self.assertTrue(torch.allclose(
            hmm.markov.transition_probs[order][:, order], hmm_reference.markov.transition_probs,
            atol=0.01, rtol=0
        ))

        self.assertTrue(torch.allclose(
            hmm.emission.means[order], hmm_reference.emission.means,
            atol=0.05, rtol=0
        ))

        self.assertTrue(torch.allclose(
            hmm.emission.covars[order], hmm_reference.emission.covars,
            atol=0.02, rtol=0
        ))

    def test_by_restore_discrete(self):
        """
        Test Discrete HMM with by restoring an existing HMM.
        """
        config = {
            'num_states': 3,
            'output': 'discrete',
            'output_num_states': 4
        }

        hmm_reference = HMM(**config)

        hmm_reference.markov.initial_probs.set_(torch.as_tensor(
            [0.75, 0.25, 0], dtype=torch.float
        ))
        hmm_reference.markov.transition_probs.set_(torch.as_tensor([
            [0.9, 0.1, 0],
            [0.5, 0, 0.5],
            [0, 1, 0]
        ], dtype=torch.float))

        hmm_reference.emission.probabilities.set_(torch.as_tensor([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0.5, 0.5]
        ]))

        torch.manual_seed(21)
        sequences = hmm_reference.sample(16536, 16)

        hmm = HMM(**config)
        hmm.fit(sequences, epochs=100, eps=1e-7, patience=3)

        order = hmm.markov.initial_probs.argsort(descending=True)

        self.assertTrue(torch.allclose(
            hmm.markov.initial_probs[order], hmm_reference.markov.initial_probs,
            atol=0.01, rtol=0
        ))

        self.assertTrue(torch.allclose(
            hmm.markov.transition_probs[order][:, order], hmm_reference.markov.transition_probs,
            atol=0.01, rtol=0
        ))

        self.assertTrue(torch.allclose(
            hmm.emission.probabilities[order], hmm_reference.emission.probabilities,
            atol=0.05, rtol=0
        ))


if __name__ == '__main__':
    unittest.main()
