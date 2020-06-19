import unittest
import numpy as np
import torch
import torch.distributions as dist
from sklearn.mixture import GaussianMixture
from pycave.bayes._internal.utils import log_normal, log_responsibilities, packed_get_last, \
    packed_drop_last

class TestUtils(unittest.TestCase):
    """
    Test case to test the correctness of PyCave's utility functions.
    """

    def test_log_normal_diag(self):
        """
        Test the log-normal probabilities for diagonal covariance.
        """
        N = 100
        S = 50
        D = 10

        means = torch.randn(S, D)
        covs = torch.rand(S, D)
        x = torch.randn(N, D)

        distributions = [
            dist.MultivariateNormal(means[i], torch.diag(covs[i]))
            for i in range(S)
        ]

        expected = []
        for item in x:
            e_item = []
            for d in distributions:
                e_item.append(d.log_prob(item))
            expected.append(e_item)
        expected = torch.as_tensor(expected)

        predicted = log_normal(x, means, covs)

        self.assertTrue(
            torch.allclose(expected, predicted, atol=1e-03, rtol=1e-05)
        )

    def test_log_normal_diag_shared(self):
        """
        Test the log-normal probabilities for shared diagonal covariance.
        """
        N = 100
        S = 50
        D = 10

        means = torch.randn(S, D)
        covs = torch.rand(D)
        x = torch.randn(N, D)

        distributions = [
            dist.MultivariateNormal(means[i], torch.diag(covs))
            for i in range(S)
        ]

        expected = []
        for item in x:
            e_item = []
            for d in distributions:
                e_item.append(d.log_prob(item))
            expected.append(e_item)
        expected = torch.as_tensor(expected)

        predicted = log_normal(x, means, covs)

        self.assertTrue(
            torch.allclose(expected, predicted, atol=1e-03, rtol=1e-05)
        )

    def test_log_normal_spherical(self):
        """
        Test the log-normal probabilities for spherical covariance.
        """
        N = 100
        S = 50
        D = 10

        means = torch.randn(S, D)
        covs = torch.rand(S)
        x = torch.randn(N, D)

        distributions = [
            dist.MultivariateNormal(
                means[i], torch.diag(covs[i].clone().expand(D))
            ) for i in range(S)
        ]

        expected = []
        for item in x:
            e_item = []
            for d in distributions:
                e_item.append(d.log_prob(item))
            expected.append(e_item)
        expected = torch.as_tensor(expected)

        predicted = log_normal(x, means, covs)

        self.assertTrue(
            torch.allclose(expected, predicted, atol=1e-03, rtol=1e-05)
        )

    def test_log_responsibilities(self):
        """
        Test the log responsibilities with the help of Sklearn.
        """
        N = 16384
        S = 2048
        D = 128

        means = torch.randn(S, D)
        covs = torch.rand(S)
        x = torch.randn(N, D)
        prior = torch.rand(S)
        prior /= prior.sum()
        mixture = GaussianMixture(S, covariance_type='spherical')
        mixture.means_ = means.numpy()
        mixture.precisions_cholesky_ = np.sqrt(1 / covs.numpy())
        mixture.weights_ = prior.numpy()

        # pylint: disable=protected-access
        _, expected = mixture._estimate_log_prob_resp(x.numpy())
        expected = torch.from_numpy(expected)

        probs = log_normal(x, means, covs)
        predicted = log_responsibilities(probs, prior)

        self.assertTrue(
            torch.allclose(expected, predicted, atol=1e-03, rtol=1e-05)
        )

    def test_packed_get_last(self):
        """
        Tests whether the last item of packed sequences can be retrieved.
        """
        a = torch.as_tensor([1, 2, 3, 4])
        b = torch.as_tensor([5, 6, 7])
        c = torch.as_tensor([8, 9])
        packed = torch.nn.utils.rnn.pack_sequence([a, b, c])

        suffix = packed_get_last(packed.data, packed.batch_sizes)
        expected = torch.as_tensor([4, 7, 9])

        self.assertTrue(torch.all(suffix == expected))

    def test_packed_drop_last(self):
        """
        Tests whether the last item of packed sequences can be dropped.
        """
        a = torch.as_tensor([1, 2, 3, 4])
        b = torch.as_tensor([5, 6, 7])
        c = torch.as_tensor([8, 9])
        packed = torch.nn.utils.rnn.pack_sequence([a, b, c])

        dropped = packed_drop_last(packed.data, packed.batch_sizes)
        expected = torch.as_tensor([1, 5, 8, 2, 6, 3])

        self.assertTrue(torch.all(dropped == expected))


if __name__ == '__main__':
    unittest.main()
