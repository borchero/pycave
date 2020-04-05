import pyblaze.nn as xnn

class HMMConfig(xnn.Config):
    """
    The configuration can be used to customize a hidden Markov model. Values that do not have a
    default value must be passed to the initializer.
    """

    num_states: int
    """
    The number of states in the hidden Markov model.
    """

    output: str = 'gaussian'
    """
    The type of output that the HMM generates. Currently must be one of the following:

    * gaussian: Every state's output is a Gaussian distribution with some mean and covariance.
    * discrete: All states have a shared set of discrete output states.
    """

    output_num_states: int = 1
    """
    The number of output states. Only applies if the output is discrete. Should be given in this
    case.
    """

    output_dim: int = 1
    """
    The dimensionality of the gaussian distributions. Only applies if the output is Gaussian. Should
    be given in this case.
    """

    output_covariance: str = 'diag'
    """
    The type of covariance to use for the gaussian distributions. The same constraints as for the
    Gaussian mixture model apply (see `GMMConfig`). Only applies if the output is Gaussian.
    """

    def is_valid(self):
        return (
            self.output in ('gaussian', 'discrete') and
            self.output_covariance in ('diag', 'spherical', 'diag-shared')
        )
