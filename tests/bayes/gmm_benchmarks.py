import time
import unittest
import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from pycave.bayes import GMM, GMMConfig

class BenchmarkGMM(unittest.TestCase):
    """
    Test case to benchmark the performance of PyCave's GMM.
    """

    def setUp(self):
        if torch.cuda.is_available():
            dummy = torch.rand(1)
            _ = dummy.cuda()

    def test_compare(self):
        """
        Unit test to benchmark the performance of our implementation compared to sklearn.
        """
        config = GMMConfig(
            num_components=512,
            num_features=128,
            covariance='spherical'
        )

        samples = self.generate_samples(config, 100_000, 'cpu')
        sklearn_time = np.mean([self.train_sklearn(config, samples) for _ in range(3)])
        ours_cpu_time = np.mean([self.train_ours(config, samples) for _ in range(3)])
        ours_gpu_time = np.mean([
            self.train_ours(config, samples.cuda(), 'cuda:0') for _ in range(3)
        ])

        print(f"-------------------------------------")
        print(f"Speedup of CPU implementation: {sklearn_time / ours_cpu_time:.2f}")
        print(f"Speedup of GPU implementation: {sklearn_time / ours_gpu_time:.2f}")
        print(f"-------------------------------------")

    def test_minibatch(self):
        """
        Unit test to benchmark the performance when training on mini batches.
        """
        config = GMMConfig(
            num_components=512,
            num_features=128,
            covariance='spherical'
        )

        samples = self.generate_samples(config, 10_000_000, 'cpu')
        samples = samples.repeat(10, 1)

        total_time = np.mean([
            self.train_ours(config, samples, 'cuda:0', batch_size=750_000) for _ in range(3)
        ])

        print(f"-------------------------------------")
        print(f"Mini-batch training took: {total_time:.2f}")
        print(f"-------------------------------------")

    def generate_samples(self, config, num_samples, device):
        """
        Generates samples by initializing a random GMM with the specified configuration.
        """
        tic = time.time()

        generator = GMM(config)
        weights = torch.rand(config.num_components)
        generator.component_weights.set_(weights / weights.sum())
        generator.means.set_(torch.randn(config.num_components, config.num_features))

        if config.covariance == 'diag':
            generator.covars.set_(torch.rand(config.num_components, config.num_features))

        samples = generator.sample(num_samples).to(device)

        toc = time.time()
        print(f"Generated {num_samples:,} samples in {toc-tic:.2f} seconds.")

        return samples

    def train_sklearn(self, config, samples):
        """
        Fits an sklearn GMM as defined by the config on the given samples.
        """
        num_samples = samples.size(0)
        samples = samples.numpy()
        tic = time.time()

        gmm = GaussianMixture(
            config.num_components,
            config.covariance,
            init_params='kmeans',
            tol=1e-7 * num_samples
        )
        gmm.fit(samples)

        toc = time.time()

        print(f"Training with sklearn took {toc-tic:.2f} seconds.")
        print(f"    Number of iterations was:    {gmm.n_iter_:,}")
        print(f"    Negative log-likelihood was: {-gmm.lower_bound_:.4f}")

        return toc - tic

    def train_ours(self, config, samples, device='cpu', batch_size=None):
        """
        Trains our model and optionally initializes with a fraction of the available data.
        """
        num_samples = samples.size(0)

        tic = time.time()

        gmm = GMM(config).to(device)
        history = gmm.fit(samples, batch_size=batch_size)

        toc = time.time()

        nll = history.neg_log_likelihood[-1]

        print(f"Training with pycave took {toc-tic:.2f} seconds.")
        print(f"    Number of iterations was:    {len(history):,}")
        print(f"    Negative log-likelihood was: {nll * num_samples:.4f}")

        return toc - tic


if __name__ == '__main__':
    unittest.main()
