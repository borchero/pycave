import unittest
import numpy as np
import torch
import torch.distributions as dist
from sklearn.mixture import GaussianMixture
from pycave.bayes.utils import log_normal, log_responsibilities

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
        # pylint: disable=not-callable
        expected = torch.tensor(expected)

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
        # pylint: disable=not-callable
        expected = torch.tensor(expected)

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
        # pylint: disable=not-callable
        expected = torch.tensor(expected)

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
        mixture = GaussianMixture(S, 'spherical')
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

if __name__ == '__main__':
    unittest.main()
