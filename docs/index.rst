PyCave Documentation
====================

PyCave allows you to run traditional machine learning models on CPU, GPU, and even on multiple nodes. All models are implemented in `PyTorch <https://pytorch.org/>`_ and provide an ``Estimator`` API that is fully compatible with `scikit-learn <https://scikit-learn.org/stable/>`_.

.. image:: https://img.shields.io/pypi/v/pycave?label=version
.. image:: https://img.shields.io/pypi/l/pycave


Features
--------

- Support for GPU and multi-node training by implementing models in PyTorch and relying on `PyTorch Lightning <https://www.pytorchlightning.ai/>`_
- Mini-batch training for all models such that they can be used on huge datasets
- Well-structured implementation of models

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

This dataset consists of three clusters with 8-dimensional datapoints. If you want to fit a K-Means
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

GPU and Multi-Node training
^^^^^^^^^^^^^^^^^^^^^^^^^^^

For GPU- and multi-node training, PyCave leverages PyTorch Lightning. The hardware that training
runs on is determined by the :class:`pytorch_lightning.trainer.Trainer` class. It's
:meth:`~pytorch_lightning.trainer.Trainer.__init__` method provides various configuration options.

If you want to run K-Means with a GPU, you can pass the option ``accelerator='gpu'`` and ``devices=1`` to the estimator's
initializer:

.. code-block:: python

    estimator = KMeans(3, trainer_params=dict(accelerator='gpu', devices=1))

Similarly, if you want to train on 4 nodes simultaneously where each node has one GPU available,
you can specify this as follows:

.. code-block:: python

    estimator = KMeans(3, trainer_params=dict(num_nodes=4, accelerator='gpu', 1))

In fact, **you do not need to change anything else in your code**.


Implemented Models
^^^^^^^^^^^^^^^^^^

Currently, PyCave implements three different models. Some of these models are also available in
scikit-learn. In this case, we benchmark our implementation against their (see
:doc:`here <sites/benchmark>`).

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
