import pyblaze.nn as xnn

class MarkovModelConfig(xnn.Config):
    """
    The Markov model config can be used to customize the Markov model.
    """

    num_states: int
    """
    The number of states in the Markov model.
    """
