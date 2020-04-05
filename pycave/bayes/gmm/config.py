import pyblaze.nn as xnn

class GMMConfig(xnn.Config):
    """
    The GMM config can be used to customize the Gaussian mixture model. Values that do not have a
    default value must be passed to the initializer of the GMM.
    """

    num_components: int
    """
    The number of gaussian distributions that make up the GMM.
    """

    num_features: int
    """
    The dimensionality of the gaussian distributions.
    """

    covariance: str = 'diag'
    """
    The type of covariance to use for the gaussian distributions. Must be one of:

    * diag: Diagonal covariance for every component (parameters: `num_features * num_components`).
    * spherical: Spherical covariance for every component (parameters: `num_components`).
    * diag-shared: Shared diagonal covariance for all components (parameters: `num_features`).
    """

    def is_valid(self):
        return (
            self.covariance in ('diag', 'spherical', 'diag-shared')
        )
