import time
import unittest
import math
import numpy as np
import torch
from sklearn.mixture import GaussianMixture
from pycave.bayes import GMM

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
        config = {
            'num_components': 512,
            'num_features': 128,
            'covariance': 'spherical'
        }

        samples = self.generate_samples(config, 100_000)
        sklearn_time = np.mean([self.train_sklearn(config, samples) for _ in range(3)])
        ours_cpu_time = np.mean([self.train_ours(config, samples) for _ in range(3)])
        ours_gpu_time = np.mean([
            self.train_ours(config, samples.cuda(), gpu=True) for _ in range(3)
        ])

        print(f"-------------------------------------")
        print(f"Speedup of CPU implementation: {sklearn_time / ours_cpu_time:.2f}")
        print(f"Speedup of GPU implementation: {sklearn_time / ours_gpu_time:.2f}")
        print(f"-------------------------------------")

    def test_minibatch(self):
        """
        Unit test to benchmark the performance when training on mini batches.
        """
        config = {
            'num_components': 512,
            'num_features': 128,
            'covariance': 'spherical'
        }

        samples = self.generate_samples(config, 10_000_000)
        samples = samples.repeat(10, 1)

        total_time = np.mean([
            self.train_ours(config, samples, gpu=True, batch_size=750_000) for _ in range(3)
        ])

        print(f"-------------------------------------")
        print(f"Mini-batch training took: {total_time:.2f}")
        print(f"-------------------------------------")

    def generate_samples(self, config, num_samples):
        """
        Generates samples by initializing a random GMM with the specified configuration.
        """
        tic = time.time()

        generator = GMM(**config)
        weights = torch.rand(config.num_components)
        generator.component_weights.set_(weights / weights.sum())
        generator.gaussian.means.set_(torch.randn(config.num_components, config.num_features))

        if config.covariance == 'diag':
            generator.gaussian.covars.set_(torch.rand(config.num_components, config.num_features))

        samples = generator.sample(num_samples)

        toc = time.time()
        print(f"Generated {num_samples:,} samples in {toc-tic:.2f} seconds.")

        return samples

    def train_sklearn(self, config, samples):
        """
        Fits an sklearn GMM as defined by the config on the given samples.
        """
        samples = samples.numpy()
        tic = time.time()

        gmm = GaussianMixture(
            config['num_components'],
            covariance_type=config['covariance'],
            init_params='random',
            tol=1e-5
        )
        gmm.fit(samples)

        toc = time.time()

        print(f"Training with sklearn took {toc-tic:.2f} seconds.")
        print(f"    Number of iterations was:    {gmm.n_iter_:,}")
        print(f"    Negative log-likelihood was: {-gmm.lower_bound_:.4f}")

        return toc - tic

    def train_ours(self, config, samples, gpu=False, batch_size=None):
        """
        Trains our model and optionally initializes with a fraction of the available data.
        """
        tic = time.time()

        gmm = GMM(**config)
        if batch_size is not None:
            samples = samples.chunk(int(math.ceil(samples.size(0) / batch_size)))
        history = gmm.fit(samples, gpu=gpu)

        toc = time.time()

        nll = history.neg_log_likelihood[-1]

        print(f"Training with pycave took {toc-tic:.2f} seconds.")
        print(f"    Number of iterations was:    {len(history):,}")
        print(f"    Negative log-likelihood was: {nll:.4f}")

        return toc - tic


if __name__ == '__main__':
    unittest.main()
