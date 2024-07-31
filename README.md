# TorchGMM

<!-- ![PyPi](https://img.shields.io/pypi/v/torchgmm?label=version)
![License](https://img.shields.io/pypi/l/torchgmm) -->

TorchGMM allows to run Gaussian Mixture Models on single or multiple CPUs/GPUs.
The repository is a fork from [PyCave](https://github.com/borchero/pycave) and [LightKit](https://github.com/borchero/lightkit), two amazing packages developed by [Olivier Borchert](https://github.com/borchero) that are not being maintained anymore.
While PyCave implements additional models such as Markov Chains, TorchGMM focuses only on Gaussian Mixture Models.

The models are implemented in [PyTorch](https://pytorch.org/) and [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/), and provide an `Estimator` API
that is fully compatible with [scikit-learn](https://scikit-learn.org/stable/).

For Gaussian mixture model, TorchGMM allows for 100x speed ups when using a GPU and enables to train
on markedly larger datasets via mini-batch training. The full suite of benchmarks run to compare
TorchGMM models against scikit-learn models is available on the
[documentation website](https://pycave.borchero.com/sites/benchmark.html).

## Features

- Support for GPU and multi-node training by implementing models in PyTorch and relying on
  [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/)
- Mini-batch training for all models such that they can be used on huge datasets
- Well-structured implementation of models

  - High-level `Estimator` API allows for easy usage such that models feel and behave like in
    scikit-learn
  - Medium-level `LightingModule` implements the training algorithm
  - Low-level PyTorch `Module` manages the model parameters

## Installation

TorchGMM is available via `pip`:

```bash
pip install torchgmm
```

If you are using [Poetry](https://python-poetry.org/):

```bash
poetry add torchgmm
```

## Usage

If you've ever used scikit-learn, you'll feel right at home when using TorchGMM. First, let's create
some artificial data to work with:

```python
import torch

X = torch.cat([
    torch.randn(10000, 8) - 5,
    torch.randn(10000, 8),
    torch.randn(10000, 8) + 5,
])
```

This dataset consists of three clusters with 8-dimensional datapoints. If you want to fit a K-Means
model, to find the clusters' centroids, it's as easy as:

```python
from torchgmm.clustering import KMeans

estimator = KMeans(3)
estimator.fit(X)

# Once the estimator is fitted, it provides various properties. One of them is
# the `model_` property which yields the PyTorch module with the fitted parameters.
print("Centroids are:")
print(estimator.model_.centroids)
```

Due to the high-level estimator API, the usage for all machine learning models is similar. The API
documentation provides more detailed information about parameters that can be passed to estimators
and which methods are available.

### GPU and Multi-Node training

For GPU- and multi-node training, TorchGMM leverages PyTorch Lightning. The hardware that training
runs on is determined by the
[Trainer](https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.trainer.trainer.html#pytorch_lightning.trainer.trainer.Trainer)
class. It's
[**init**](https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.trainer.trainer.html#pytorch_lightning.trainer.trainer.Trainer.__init__)
method provides various configuration options.

If you want to run K-Means with a GPU, you can pass the options `accelerator='gpu'` and `devices=1`
to the estimator's initializer:

```python
estimator = KMeans(3, trainer_params=dict(accelerator='gpu', devices=1))
```

Similarly, if you want to train on 4 nodes simultaneously where each node has one GPU available,
you can specify this as follows:

```python
estimator = KMeans(3, trainer_params=dict(num_nodes=4, accelerator='gpu', devices=1))
```

In fact, **you do not need to change anything else in your code**.

### Implemented Models

Currently, TorchGMM implements two different models:

- [GaussianMixture](https://pycave.borchero.com/sites/generated/bayes/gmm/pycave.bayes.GaussianMixture.html)
- [K-Means](https://pycave.borchero.com/sites/generated/clustering/kmeans/pycave.clustering.KMeans.html)

## License

TorchGMM is licensed under the [MIT License](https://github.com/marcovarrone/torchgmm/blob/main/LICENSE).
