PyCave Documentation
====================

.. image:: https://img.shields.io/pypi/v/pycave?label=version

PyCave provides well-known machine learning models for the usage with large-scale datasets. This is
achieved by leveraging PyTorch's capability to easily perform computations on a GPU as well as
implementing batch-wise training for all models.

As a result, PyCave's models are able to work with datasets orders of magnitudes larger than
datasets that are commonly used with Sklearn. At the same time, PyCave provides an API that is very
familiar both to users of Sklearn and PyTorch.

Internally, PyCave's capabilities are heavily supported by `PyBlaze <https://github.com/borchero/pyblaze>`_ which enables seamless batch-wise GPU training without additional code. 

Models
------

PyCave currently includes the following models:

* :code:`pycave.bayes.GMM`: `Gaussian Mixture Models <https://en.wikipedia.org/wiki/Mixture_model#Gaussian_mixture_model>`_
* :code:`pycave.bayes.HMM`: `Hidden Markov Models <https://en.wikipedia.org/wiki/Hidden_Markov_model>`_ with discrete and Gaussian emissions
* :code:`pycave.bayes.MarkovModel`: `Markov Models <https://en.wikipedia.org/wiki/Markov_model>`_

All of these models can be trained on a (single) GPU and using batches of data.

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
