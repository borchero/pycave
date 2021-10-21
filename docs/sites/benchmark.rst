Benchmarks
==========

In order to evaluate the runtime performance of PyCave, we run an exhaustive set of experiments to
compare against the implementation found in scikit-learn. Evaluations are run at varying dataset
sizes.

All benchmarks are run on an instance with a Intel Xeon E5-2630 v4 CPU (2.2 GHz). We use at most 4
cores and 60 GiB of memory. Also, there is a single GeForce GTX 1080 Ti GPU (11 GiB memory)
available. For the performance measures, each benchmark is run at least 5 times.

Gaussian Mixture
----------------

Setup
^^^^^

For measuring the performance of fitting a Gaussian mixture model, we fix the number of iterations
after initialization to 100 to not measure any variances in the convergence criterion. Further, we
perform random initialization to not measure the time taken for running K-Means (benchmarks for
K-Means are available below).

Results
^^^^^^^

.. list-table:: Training Duration for Spherical Covariance (``[num_datapoints, num_features] -> num_components``)
    :header-rows: 1
    :stub-columns: 1
    :widths: 3 2 2 2 2 2

    * - 
      - Scikit-Learn
      - PyCave CPU (full)
      - PyCave CPU (batches)
      - PyCave GPU (full)
      - PyCave GPU (batches)
    * - ``[10k, 8] -> 4``
      - 
      - 
      - 
      - 
      - 
    * - ``[100k, 32] -> 16``
      - 
      - 
      - 
      - 
      - 
    * - ``[1M, 64] -> 64``
      - 
      - 
      - 
      - 
      - 

.. list-table:: Training Duration for Diagonal Covariance (``[num_datapoints, num_features] -> num_components``)
    :header-rows: 1
    :stub-columns: 1
    :widths: 3 2 2 2 2 2

    * - 
      - Scikit-Learn
      - PyCave CPU (full)
      - PyCave CPU (batches)
      - PyCave GPU (full)
      - PyCave GPU (batches)
    * - ``[10k, 8] -> 4``
      - 
      - 
      - 
      - 
      - 
    * - ``[100k, 32] -> 16``
      - 
      - 
      - 
      - 
      - 
    * - ``[1M, 64] -> 64``
      - 
      - 
      - 
      - 
      - 

.. list-table:: Training Duration for Tied Covariance (``[num_datapoints, num_features] -> num_components``)
    :header-rows: 1
    :stub-columns: 1
    :widths: 3 2 2 2 2 2

    * - 
      - Scikit-Learn
      - PyCave CPU (full)
      - PyCave CPU (batches)
      - PyCave GPU (full)
      - PyCave GPU (batches)
    * - ``[10k, 8] -> 4``
      - 
      - 
      - 
      - 
      - 
    * - ``[100k, 32] -> 16``
      - 
      - 
      - 
      - 
      - 
    * - ``[1M, 64] -> 64``
      - 
      - 
      - 
      - 
      - 

.. list-table:: Training Duration for Full Covariance (``[num_datapoints, num_features] -> num_components``)
    :header-rows: 1
    :stub-columns: 1
    :widths: 3 2 2 2 2 2

    * - 
      - Scikit-Learn
      - PyCave CPU (full)
      - PyCave CPU (batches)
      - PyCave GPU (full)
      - PyCave GPU (batches)
    * - ``[10k, 8] -> 4``
      - 
      - 
      - 
      - 
      - 
    * - ``[100k, 32] -> 16``
      - 
      - 
      - 
      - 
      - 
    * - ``[1M, 64] -> 64``
      - 
      - 
      - 
      - 
      - 

Summary
^^^^^^^


K-Means
-------

Setup
^^^^^

For the scikit-learn implementation, we use Lloyd's algorithm instead of Elkan's algorithm to have
a useful comparison with PyCave (which implements Lloyd's algorithm).

Further, we fix the number of iterations after initialization to 100 to not measure any variances
in the convergence criterion.

Results
^^^^^^^

.. list-table:: Training Duration for Random Initialization (``[num_datapoints, num_features] -> num_clusters``)
    :header-rows: 1
    :stub-columns: 1
    :widths: 3 2 2 2 2 2

    * - 
      - Scikit-Learn
      - PyCave CPU (full)
      - PyCave CPU (batches)
      - PyCave GPU (full)
      - PyCave GPU (batches)
    * - ``[10k, 8] -> 4``
      - **13 ms**
      - 412 ms
      - 797 ms
      - 387 ms
      - 2.1 s
    * - ``[100k, 32] -> 16``
      - **311 ms**
      - 2.1 s
      - 3.4 s
      - 707 ms
      - 2.5 s
    * - ``[1M, 64] -> 64``
      - 10.0 s
      - 73.6 s
      - 58.1 s
      - **8.2 s**
      - 10.0 s
    * - ``[10M, 128] -> 128``
      - 254 s
      - --
      - --
      - --
      - **133 s**

.. list-table:: Training Duration for K-Means++ Initialization (``[num_datapoints, num_features] -> num_clusters``)
    :header-rows: 1
    :stub-columns: 1
    :widths: 3 2 2 2 2 2

    * - 
      - Scikit-Learn
      - PyCave CPU (full)
      - PyCave CPU (batches)
      - PyCave GPU (full)
      - PyCave GPU (batches)
    * - ``[10k, 8] -> 4``
      - **15 ms**
      - 170 ms
      - 930 ms
      - 431 ms
      - 2.4 s
    * - ``[100k, 32] -> 16``
      - **542 ms**
      - 2.3 s
      - 4.3 s
      - 840 ms
      - 3.2 s
    * - ``[1M, 64] -> 64``
      - 25.3 s
      - 93.4 s
      - 83.7 s
      - **13.1 s**
      - 17.1 s
    * - ``[10M, 128] -> 128``
      - 827 s
      - --
      - --
      - --
      - **369 s**

Summary
^^^^^^^

As it turns out, it is really hard to outperform the implementation found in scikit-learn.
Especially if little data is available, the overhead of PyTorch and PyTorch Lightning renders
PyCave comparatively slow. However, as more data is available, PyCave starts to become relatively
faster and, when leveraging the GPU, it finally outperforms scikit-learn for a dataset size of 1M
datapoints. Nonetheless, the improvement is marginal.
