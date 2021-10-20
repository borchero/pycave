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
----------------

K-Means
-------

Setup
^^^^^

For the scikit-learn implementation, we use Lloyd's algorithm instead of Elkan's algorithm to have
a useful comparison with PyCave (which implements Lloyd's algorithm).

Results
^^^^^^^

.. list-table:: K-Means Training Duration (Random Initialization)
    :header-rows: 1
    :stub-columns: 1

    * - 
      - Scikit-Learn
      - PyCave CPU (full data)
      - PyCave CPU (mini-batches)
      - PyCave GPU (full data)
      - PyCave GPU (mini-batches)
    * - 10k datapoints, 8 features, 4 clusters
      - 12 ms
      - 416 ms
      - 830 ms
      - 194 ms
      - 975 ms
    * - 100k datapoints, 32 features, 16 clusters
      - 309 ms
      - 2.1 s
      - 3.4 s
      - 612 ms
      - 719 ms
    * - 1M datapoints, 64 features, 64 clusters
      - 10.1 s
      - 91.9 s
      - 59.0 s
      - 1.8 s
      - 2.2 s
    * - 10M datapoints, 128 features, 128 clusters
      - 
      - --
      - --
      - --
      - 29.6 s

.. list-table:: K-Means Training Duration (K-Means++ Initialization)
    :header-rows: 1
    :stub-columns: 1

    * - 
      - Scikit-Learn
      - PyCave CPU (full data)
      - PyCave CPU (mini-batches)
      - PyCave GPU (full data)
      - PyCave GPU (mini-batches)
    * - 10k datapoints, 8 features, 4 clusters
      - 17 ms
      - 134 ms
      - 886 ms
      - 142 ms
      - 1.1 s
    * - 100k datapoints, 32 features, 16 clusters
      - 528 ms
      - 2.2 s
      - 4.2 s
      - 559 ms
      - 3.2 s
    * - 1M datapoints, 64 features, 64 clusters
      - 24.9 s
      - 115 s
      - 81.2 s
      - 5.2 s
      - 9.1 s
    * - 10M datapoints, 128 features, 128 clusters
      - 
      - --
      - --
      - --
      - 291 s

As it turns out, it is really hard to outperform scikit-learn. Especially if little data is
available, the overhead of PyTorch and PyTorch Lightning renders PyCave comparatively slow.
However, as more data is available, PyCave starts to become relatively faster and, when leveraging
the GPU, it finally outperforms scikit-learn for a dataset size of 1M datapoints.
