# PyCave

![PyPi](https://img.shields.io/pypi/v/pycave?label=version)
![License](https://img.shields.io/pypi/l/pycave)

PyCave allows you to run traditional machine learning models on CPU, GPU, and even on multiple
nodes. All models are implemented in [PyTorch](https://pytorch.org/) and provide an `Estimator` API
that is fully compatible with [scikit-learn](https://scikit-learn.org/stable/).

For Gaussian mixture model, PyCave allows for 100x speed ups when using a GPU and enables to train
on markedly larger datasets via mini-batch training. The full suite of benchmarks run to compare
PyCave models against scikit-learn models is available on the
[documentation website](https://pycave.borchero.com/sites/benchmark.html).

_PyCave version 3 is a complete rewrite of PyCave which is tested much more rigorously, depends on
well-maintained libraries and is tuned for better performance. While you are, thus, highly
encouraged to upgrade, refer to [pycave-v2.borchero.com](https://pycave-v2.borchero.com) for
documentation on PyCave 2._

## Features

- Support for GPU and multi-node training by implementing models in PyTorch and relying on
  [PyTorch Lightning](https://www.pytorchlightning.ai/)
- Mini-batch training for all models such that they can be used on huge datasets
- Well-structured implementation of models

  - High-level `Estimator` API allows for easy usage such that models feel and behave like in
    scikit-learn
  - Medium-level `LightingModule` implements the training algorithm
  - Low-level PyTorch `Module` manages the model parameters

## Installation

PyCave is available via `pip`:

```bash
pip install pycave
```

If you are using [Poetry](https://python-poetry.org/):

```bash
poetry add pycave
```

## Usage

If you've ever used scikit-learn, you'll feel right at home when using PyCave. First, let's create
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
from pycave.clustering import KMeans

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

For GPU- and multi-node training, PyCave leverages PyTorch Lightning. The hardware that training
runs on is determined by the
[Trainer](https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.trainer.trainer.html#pytorch_lightning.trainer.trainer.Trainer)
class. It's
[**init**](https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.trainer.trainer.html#pytorch_lightning.trainer.trainer.Trainer.__init__)
method provides various configuration options.

If you want to run K-Means with a GPU, you can pass the option `gpus=1` to the estimator's
initializer:

```python
estimator = KMeans(3, trainer_params=dict(gpus=1))
```

Similarly, if you want to train on 4 nodes simultaneously where each node has one GPU available,
you can specify this as follows:

```python
estimator = KMeans(3, trainer_params=dict(num_nodes=4, gpus=1))
```

In fact, **you do not need to change anything else in your code**.

### Implemented Models

Currently, PyCave implements three different models:

- [GaussianMixture](https://pycave.borchero.com/sites/generated/bayes/gmm/pycave.bayes.GaussianMixture.html)
- [MarkovChain](https://pycave.borchero.com/sites/generated/bayes/markov_chain/pycave.bayes.MarkovChain.html)
- [K-Means](https://pycave.borchero.com/sites/generated/clustering/kmeans/pycave.clustering.KMeans.html)

## License

PyCave is licensed under the [MIT License](https://github.com/borchero/pycave/blob/master/LICENSE).
