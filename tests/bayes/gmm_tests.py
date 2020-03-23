import unittest
import numpy as np
import torch
from pycave.bayes import GMM, GMMConfig

class TestGMM(unittest.TestCase):
    """
    Test case to test the correctness of PyCave's GMM.
    """

    def test_by_restore_spherical(self):
        """
        Test GMM with spherical covariance by restoring an existing GMM.
        """
        config = GMMConfig(
            num_components=5, num_features=2, covariance='spherical'
        )

        # pylint: disable=not-callable
        gmm_reference = GMM(config)
        gmm_reference.component_weights.set_(torch.tensor(
            [0.1, 0.15, 0.2, 0.25, 0.3]
        ))
        gmm_reference.means.set_(torch.tensor([
            [-4, 1], [0, 1], [4, 1], [-3, -5], [3, -5]
        ], dtype=torch.float))
        gmm_reference.covars.set_(torch.tensor(
            [1, 1.5, 1, 1.5, 0.25]
        ))

        torch.manual_seed(42)
        data = gmm_reference.sample(16384)

        gmm = GMM(config)
        np.random.seed(42)
        gmm.reset_parameters(data)
        gmm.fit(data, max_iter=1000)

        order = gmm.component_weights.argsort()

        self.assertTrue(
            torch.allclose(
                gmm_reference.component_weights,
                gmm.component_weights[order],
                atol=0.01,
                rtol=0
            )
        )

        self.assertTrue(
            torch.allclose(
                gmm_reference.means,
                gmm.means[order],
                atol=0.1,
                rtol=0
            )
        )

        self.assertTrue(
            torch.allclose(
                gmm_reference.covars,
                gmm.covars[order],
                atol=0.1,
                rtol=0
            )
        )


if __name__ == '__main__':
    unittest.main()
