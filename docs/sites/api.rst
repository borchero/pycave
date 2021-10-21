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


Utilities
---------

Base Classes
^^^^^^^^^^^^

.. currentmodule:: pycave.core
.. autosummary::
    :toctree: generated/core
    :nosignatures:
    :caption: Base Classes
    :template: classes/no_inherited_methods.rst

    Estimator

    :template: classes/mixin.rst

    PredictorMixin
    TransformerMixin

    :template: classes/only_methods.rst

    ConfigModule


Data Loading
^^^^^^^^^^^^

.. currentmodule:: pycave.data
.. autosummary::
    :toctree: generated/data
    :nosignatures:
    :caption: Data Loading
    :template: classes/no_members.rst

    TensorDataLoader
    TensorBatchSampler
    DistributedTensorBatchSampler
    UnrepeatedDistributedTensorBatchSampler


Types
^^^^^

.. currentmodule:: pycave
.. autosummary::
    :toctree: generated/types
    :nosignatures:
    :caption: Types
    :template: classes/type_alias.rst

    ~data.TabularData
    ~data.SequenceData
    ~bayes.core.CovarianceType
    ~bayes.gmm.types.GaussianMixtureInitStrategy
    ~clustering.kmeans.types.KMeansInitStrategy
