PyCave Documentation
====================

PyCave allows you to run traditional machine learning models on CPU, GPU, TPU and even on multiple nodes. All models are implemented in `PyTorch <https://pytorch.org/>`_ and provide an ``Estimator`` API that is fully compatible with `scikit-learn <https://scikit-learn.org/stable/>`_.

.. image:: https://img.shields.io/pypi/v/pycave?label=version
.. image:: https://img.shields.io/pypi/l/pycave


Features
--------

- Support for GPU, TPU, and multi-node training by implementing models in PyTorch and relying on `PyTorch Ligthning <https://www.pytorchlightning.ai/>`_
- Mini-batch training for all models such that they can be used on huge datasets
- Highly structured implementation of models

  - High-level ``Estimator`` API allows for easy usage such that models feel and behave like in
    scikit-learn
  - Medium-level ``LightingModule`` implements the training algorithm
  - Low-level PyTorch ``Module`` manages the model parameters


Installation
------------

PyCave is available via ``pip``:

.. code-block:: python

    pip install pycave

If you are using `Poetry <https://python-poetry.org/>`_:

.. code-block:: python

    poetry add pycave


Usage
-----

If you've ever used scikit-learn, you'll feel right at home when using PyCave. First, let's create
some artificial data to work with:

.. code-block:: python

    import torch

    X = torch.cat([
        torch.randn(10000, 8) - 5,
        torch.randn(10000, 8),
        torch.randn(10000, 8) + 5,
    ])

This dataset consists of three clusters with 8-dimensional datapoints. If you want to fit a k-means
model, to find the clusters' centroids, it's as easy as:

.. code-block:: python

    from pycave.clustering import KMeans

    estimator = KMeans(3)
    estimator.fit(X)

    # Once the estimator is fitted, it provides various properties. One of them is
    # the `model_` property which yields the PyTorch module with the fitted parameters.
    print("Centroids are:")
    print(estimator.model_.centroids)

Due to the high-level estimator API, the usage for all machine learning models is similar. The API
documentation provides more detailed information about parameters that can be passed to estimators
and which methods are available.


Implemented Models
^^^^^^^^^^^^^^^^^^

.. currentmodule:: pycave

.. autosummary::
    :nosignatures:

    ~bayes.GaussianMixture
    ~bayes.MarkovChain
    ~clustering.KMeans

Reference
---------

.. toctree::
   :maxdepth: 2

   sites/benchmark
   sites/api

Index
^^^^^

- :ref:`genindex`
