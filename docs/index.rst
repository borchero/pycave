PyCave Documentation
====================

.. image:: https://img.shields.io/pypi/v/pycave?label=version

PyCave provides well-known machine learning models with strong GPU acceleration in PyTorch. Its
goal is not to provide a comprehensive collection of models or neural network layers, but rather
complement other open-source libraries.

Features
--------

PyCave currently includes the following models to be run on the GPU:

* `Gaussian Mixture Models <https://en.wikipedia.org/wiki/Mixture_model#Gaussian_mixture_model>`_, optionally trained via mini-batches if the GPU memory is too small to fit the data. Mini-batch training should *not* impact convergence. Initialization is performed using K-means, optionally on a subset of the data as it is comparatively slow.
* `Markov Models <https://en.wikipedia.org/wiki/Markov_model>`_ able to learn transition probabilities from a sequence of discrete states.
* `Hidden Markov Models <https://en.wikipedia.org/wiki/Hidden_Markov_model>`_ with discrete and Gaussian emissions.

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Guides

   guides/quickstart


.. toctree::
   :glob:
   :maxdepth: 2
   :caption: Package Reference

   modules/bayes

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`

`View on GitHub <https://github.com/borchero/pycave>`_
