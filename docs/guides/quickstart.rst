Quickstart
==========

Using PyCave is really intuitive as its API closely aligns with Sklearn and PyTorch.

Installation
------------

PyCave is available on PyPi and can simply be installed as follows:

>>> pip install pycave

Example
-------

In order to train a Gaussian Mixture Model, you can initialize it as follows:

.. code-block:: python

    from pycave.bayes import GMM
    gmm = GMM(num_components=16, num_features=32, covariance='spherical')

Consider now that you have some data `data` represented by a `torch.Tensor` with size `(10000, 32)`.

If the data is very far from normal, you might consider initializing the GMM using K-Means. *Note
that this step is optional and deteriorates the performance as it needs to be run on the CPU.*

.. code-block:: python

    gmm.reset_parameters(data)

Subsequently, the GMM can be fit to the data:

.. code-block:: python

    history = gmm.fit(data)

*If a GPU is available, this call will automatically use it.* (If this behavior is undesired, pass
:code:`gpu=False` as keyword argument to the :code:`fit` method.) The returned history object
carries, amongst others, the per-datapoint log-likelihood of the fitted model over the course of the
EM-algorithm:

.. code-block:: python

    history.neg_log_likelihood

In case the data to train on is too large to fit on the GPU (or even in RAM), training can also be
done in batches. For that, any iterable of batches of datapoints can be passed to the :code:`fit`
method. Commonly you would use a PyTorch data loader for that but it is also possible to just pass
a list of two-dimensional tensors.

The fitted GMM now provides instance methods for inference:

* :code:`gmm.evaluate` computes the negative log-likelihood of some data.
* :code:`gmm.predict` returns the probability distribution for some data and all components - an `argmax` operation yields the most likely components for the data.
* :code:`gmm.sample` samples a given number of samples from the GMM.

Consult the package reference for GMMs for more information about the GMM.
