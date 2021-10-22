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
after initialization to 100 to not measure any variances in the convergence criterion. For
initialization, we further set the known means that were used to generate data to not run into
issues of degenerate covariance matrices. Thus, all benchmarks essentially measure the performance
after K-means initialization has been run. Benchmarks for K-means itself are listed below.

Results
^^^^^^^

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
      - **352 ms**
      - 649 ms
      - 3.9 s
      - 358 ms
      - 3.6 s
    * - ``[100k, 32] -> 16``
      - 18.4 s
      - 4.3 s
      - 10.0 s
      - **527 ms**
      - 3.9 s
    * - ``[1M, 64] -> 64``
      - 730 s
      - 196 s
      - 284 s
      - **7.7 s**
      - 15.3 s

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
      - 699 ms
      - 570 ms
      - 3.6 s
      - **356 ms**
      - 3.3 s
    * - ``[100k, 32] -> 16``
      - 72.2 s
      - 12.1 s
      - 16.1 s
      - **919 ms**
      - 3.8 s
    * - ``[1M, 64] -> 64``
      - --
      - --
      - --
      - --
      - **63.4 s**

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
      - 1.1 s
      - 679 ms
      - 4.1 s
      - **648 ms**
      - 4.4 s
    * - ``[100k, 32] -> 16``
      - 110 s
      - 13.5 s
      - 21.2 s
      - **2.4 s**
      - 7.8 s

Summary
^^^^^^^

PyCave's implementation of the Gaussian mixture model is markedly more efficient than the one found
in scikit-learn. Even on the CPU, PyCave outperforms scikit-learn significantly at a 100k
datapoints already. When moving to the GPU, however, PyCave unfolds its full potential and yields
speed ups at around 100x. For larger datasets, mini-batch training is the only alternative. PyCave
fully supports that while the training is approximately twice as large as when training using the
full data. The reason for this is that the M-step of the EM algorithm needs to be split across
epochs, which, in turn, requires to replay the E-step.


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
