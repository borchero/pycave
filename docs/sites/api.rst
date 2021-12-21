API Reference
=============

Bayesian Models
---------------

.. currentmodule:: pycave.bayes

Gaussian Mixture
^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/bayes/gmm
    :nosignatures:
    :caption: Bayesian Models
    
    GaussianMixture

    :template: classes/pytorch_module.rst

    ~gmm.GaussianMixtureModel
    ~gmm.GaussianMixtureModelConfig


Markov Chain
^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/bayes/markov_chain
    :nosignatures:
    
    MarkovChain

    :template: classes/pytorch_module.rst

    ~markov_chain.MarkovChainModel
    ~markov_chain.MarkovChainModelConfig


Clustering Models
-----------------

.. currentmodule:: pycave.clustering

K-Means
^^^^^^^

.. autosummary::
    :toctree: generated/clustering/kmeans
    :nosignatures:
    :caption: Clustering Models

    KMeans

    :template: classes/pytorch_module.rst

    ~kmeans.KMeansModel
    ~kmeans.KMeansModelConfig


Utility Types
-------------

.. currentmodule:: pycave
.. autosummary::
    :toctree: generated/types
    :nosignatures:
    :caption: Types
    :template: classes/type_alias.rst

    ~bayes.markov_chain.types.SequenceData
    ~bayes.core.CovarianceType
    ~bayes.gmm.types.GaussianMixtureInitStrategy
    ~clustering.kmeans.types.KMeansInitStrategy
