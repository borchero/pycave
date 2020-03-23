Quickstart
==========

Making use of PyCave is really easy since it builds directly on PyTorch and does not need to be built locally.

Installation
------------

PyCave is available on PyPi and can simply be installed as follows:

>>> pip install pycave

Example
-------

PyCave's interface is oriented towards Sklearn. In order to train a Gaussian Mixture Model, you can
initialize it as follows:

.. code-block:: python

    from pycave.bayes import GMM
    gmm = GMM(num_components=16, num_features=32, covariance='spherical')

Consider now that you have some data `data` represented by a `torch.Tensor` with size `(10000, 32)`.

If the data is very far from normal, you might consider initializing the GMM using K-Means. *Note that this step is optional and deteriorates the performance as it needs to be run on the CPU.*

.. code-block:: python

    gmm.reset_parameters(data)

Subsequently, the GMM can be fit to the data:

.. code-block:: python

    history = gmm.fit(data)

The returned history object carries, amongst others, the per-datapoint log-likelihood of the fitted model over the course of the EM-algorithm:

.. code-block:: python

    history.neg_log_likelihood

The fitted GMM now provides instance methods for inference:

* `gmm.evaluate` computes the negative log-likelihood of some data.
* `gmm.predict` returns the indices of most likely components for some data.
* `gmm.sample` samples a given number of samples from the GMM.

Consult the package reference for GMMs for more information about the GMM.
