# PyCave

![PyPi](https://img.shields.io/pypi/v/pycave?label=version)

PyCave provides well-known machine learning models with strong GPU acceleration in PyTorch. Its
goal is not to provide a comprehensive collection of models or neural network layers, but rather
complement other open-source libraries. All models are implemented based on [PyBlaze](https://github.com/borchero/pyblaze),
enabling straightforward training on GPUs and a neural-network-like training interface.

## Features

PyCave currently includes the following models to be run on the GPU:

* `pycave.bayes.GMM`: [Gaussian Mixture Models](https://en.wikipedia.org/wiki/Mixture_model#Gaussian_mixture_model), optionally trained via mini-batches if the GPU memory is too small to fit the data. Mini-batch training should *not* impact convergence. Initialization is performed using K-means, optionally on a subset of the data as it is comparatively slow.
* `pycave.bayes.MarkovModel`: [Markov Models](https://en.wikipedia.org/wiki/Markov_model) able to learn transition probabilities from a sequence of discrete states.
* `pycave.bayes.HMM`: [Hidden Markov Models](https://en.wikipedia.org/wiki/Hidden_Markov_model) with discrete and Gaussian outputs.

## Installation

PyCave is available on PyPi and can simply be installed as follows:

```bash
pip install pycave
```

## Quickstart

Using PyCave is really easy if you are familiar with PyTorch and/or Sklearn. PyCave provides both, an interface closely oriented towards PyTorch's neural network training loop and Sklearn's minimal interface for estimators. Both of these features are essentially provided by [PyBlaze](https://github.com/borchero/pyblaze).

A common example is training a GMM. You can initialize it as follows and simply fit it from a `torch.Tensor`:

```python
from pycave.bayes import GMM

gmm = GMM(num_components=100, num_features=32, covariance='spherical')
gmm.fit(data_tensor)
```

You can then use the GMM's instance methods for inference:

* `gmm.evaluate` computes the negative log-likelihood of some data.
* `gmm.predict` returns the indices of most likely components for some data.
* `gmm.sample` samples a given number of samples from the GMM.

## Benchmarks

In order to demonstrate the potential of PyCave, we compared the runtime of PyCave both on CPU and GPU against the runtime of Sklearn's [Gaussian Mixture Model](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html).

We train on 100k 128-dimensional datapoints sampled from a "ground truth" GMM with 512 components. PyCave's GMM and Sklearn should then minimize the negative log-likelihood (NLL) of the data. While PyCave's GMM worked well with random initialization, Sklearn required (a single-pass) K-Means initialization to yield useful results. In both cases, the GMM converged when the per-datapoint NLL was below 1e-7.

| Implementation | Training Time | Speedup Compared to Sklearn |
| --- | --- | --- |
| Sklearn (CPU) | 114.41s | **x1** |
| PyCave (CPU) | 32.07s | **x3.57** |
| PyCave (GPU) | 0.27s | **x425.19** |

By moving to PyCave's GPU implementation of GMMs, you can therefore expects speedups by a factor of hundreds.

For huge datasets, PyCave's GMM also supports mini-batch training on a GPU. We run PyCave's GMM on the same kind of data as described above, yet on 100 million instead of 100k datapoints. We use a batch size of 750k to train on a GPU.

| Implementation | Training Time |
| --- | --- |
| PyCave (GPU, mini-batch) | 247.95s |

Even on this huge dataset, PyCave is able to fit the GMM in just over 4 minutes.

*We ran the benchmark on 8 Cores of an Intel Xeon E5-2630 with 2.2 GHz and a single GeForce GTX 1080 GPU with 11 GB of memory.*

## License

PyCave is licensed under the [MIT License](https://github.com/borchero/pycave/blob/master/LICENSE).
