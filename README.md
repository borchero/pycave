# PyCave

![PyPi](https://img.shields.io/pypi/v/pycave?label=version)

PyCave provides well-known machine learning models for the usage with large-scale datasets. This is
achieved by leveraging PyTorch's capability to easily perform computations on a GPU as well as
implementing batch-wise training for all models.

As a result, PyCave's models are able to work with datasets orders of magnitudes larger than
datasets that are commonly used with Sklearn. At the same time, PyCave provides an API that is very
familiar both to users of Sklearn and PyTorch.

Internally, PyCave's capabilities are heavily supported by [PyBlaze](https://github.com/borchero/pyblaze) which enables seamless batch-wise GPU training without additional code. 

## Features

PyCave currently includes the following models:

* `pycave.bayes.GMM`: [Gaussian Mixture Models](https://en.wikipedia.org/wiki/Mixture_model#Gaussian_mixture_model)
* `pycave.bayes.HMM`: [Hidden Markov Models](https://en.wikipedia.org/wiki/Hidden_Markov_model) with discrete and Gaussian emissions
* `pycave.bayes.MarkovModel`: [Markov Models](https://en.wikipedia.org/wiki/Markov_model)

All of these models can be trained on a (single) GPU and using batches of data.

## Installation

PyCave is available on PyPi and can simply be installed as follows:

```bash
pip install pycave
```

## Quickstart

A simple guide is available [in the documentation](https://pycave.borchero.com/guides/quickstart.html).

## Benchmarks

In order to demonstrate the potential of PyCave, we compared the runtime of PyCave both on CPU and GPU against the runtime of Sklearn's [Gaussian Mixture Model](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html).

We train on 100k 128-dimensional datapoints sampled from a "ground truth" GMM with 512 components. PyCave's GMM and Sklearn should then minimize the negative log-likelihood (NLL) of the data. For our GMM, convergence was reached when converging on a per-datapoint NLL of 0.01. For Sklearn, we had to set it to 1e-5. Initialization was random instead of K-Means initialization as K-Means needs to be run via Sklearn, hence on the CPU.

| Implementation | Avg. Train Duration | Speedup |
| --- | --- | --- |
| Sklearn (CPU) | 299.04s | - | 
| PyCave (CPU) | 31.29s | **x9.56** |
| PyCave (GPU) | 0.35s | **x864.36** |

By moving to PyCave's GPU implementation of GMMs, you can therefore expects speedups by a factor of hundreds.

For huge datasets, PyCave's GMM also supports mini-batch training on a GPU. We run PyCave's GMM on the same kind of data as described above, yet on 100 million instead of 100k datapoints. We use a batch size of 750k to train on a GPU.

| Implementation | Training Time |
| --- | --- |
| PyCave (GPU, mini-batch) | 373.23s |

As can be observed, the GMM training scales almost linearly with the number of datapoints (0.35s from the table above times 1,000).

*We ran the benchmark on 8 Cores of an Intel Xeon E5-2630 with 2.2 GHz and a single GeForce GTX 1080 GPU with 11 GB of memory.*

## License

PyCave is licensed under the [MIT License](https://github.com/borchero/pycave/blob/master/LICENSE).
