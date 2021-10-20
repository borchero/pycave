Benchmarks
==========

In order to evaluate the runtime performance of PyCave, we run an exhaustive set of experiments to
compare against the implementation found in scikit-learn. Evaluations are run at varying dataset
sizes.

As PyCave and scikit-learn might not need the same number of iterations for convergence, we disable the convergence criterion and set a fixed number of iterations depending on the dataset
size.

All benchmarks are run on an instance with a Intel Xeon E5-2630 v4 CPU (2.2 GHz). We use at most 4
cores and 60 GiB of memory. Also, there is a single GeForce GTX 1080 Ti GPU (11 GiB memory)
available. For the performance measures, each benchmark is run at least 5 times.

Gaussian Mixture
^^^^^^^^^^^^^^^^

K-Means
^^^^^^^

For K-Means, it is really hard to outperform scikit-learn. Especially if little data is available,
the overhead of PyTorch and PyTorch Lightning renders PyCave comparatively slow. However, as more
data is available, PyCave starts to become relatively faster and, when leveraging the GPU, it
finally outperforms scikit-learn for a dataset size of 1M datapoints.

.. list-table:: K-Means Relative Performance
    :header-rows: 1
    :stub-columns: 1

    * - 
      - Scikit-Learn
      - PyCave CPU (full data)
      - PyCave CPU (mini-batches)
      - PyCave GPU (full data)
      - PyCave GPU (mini-batches)
    * - 10k datapoints, 8 features, 4 clusters
      - **x 1** (19 ms)
      - **x 0.06** (318 ms)
      - 
      - **x 0.02** (1.1 s)
      - 
    * - 100k datapoints, 32 features, 16 clusters
      - **x 1** (899 ms)
      - **x 0.30** (3.0 s)
      - 
      - **x 0.41** (2.2 s)
      - 
    * - 1M datapoints, 64 features, 64 clusters
      - **x 1** (46.8 s)
      - **x 0.23** (204 s)
      - 
      - **x 4.22** (11.1 s)
      - 
    * - 10M datapoints, 128 features, 128 clusters
      - --
      - --
      - --
      -
      -
    * - 1T datapoints, 512 features, 1024 clusters
      - --
      - --
      - --
      - --
      - 
