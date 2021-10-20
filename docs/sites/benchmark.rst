Benchmarks
==========

In order to evaluate the runtime performance of PyCave, we run an exhaustive set of experiments to
compare against the implementation found in scikit-learn. Evaluations are run at varying dataset
sizes.

All benchmarks are run on an instance with a Intel Xeon E5-2630 v4 CPU (2.2 GHz) and 20 cores. Also, there is a single GeForce GTX 1080 Ti GPU (11 GiB memory) available. For the performance
measures, each benchmark is run at least 5 times.

Gaussian Mixture
^^^^^^^^^^^^^^^^

.. list-table:: Gaussian Mixture Relative Performance
    :header-rows: 1
    :stub-columns: 1

    * - 
      - Scikit-Learn
      - PyCave CPU (full data)
      - PyCave CPU (1k batches)
      - PyCave GPU (full data)
      - PyCave GPU (1k batches)
    * - 10k datapoints, 8 features, 4 clusters
      - **x 1** (19 ms)
      - **x 0.04** (493 ms)
      - 
      - 
      - 
    * - 100k datapoints, 32 features, 16 clusters
      - **x 1** (989 ms)
      - **x 0.27** (3.7 s)
      - 
      - 
      - 
    * - 1M datapoints, 64 features, 64 clusters
      - **x 1** (64.9 s)
      - **x 0.22** (293 s)
      - 
      - 
      - 

K-Means
^^^^^^^

.. list-table:: K-Means Relative Performance
    :header-rows: 1
    :stub-columns: 1

    * - 
      - Scikit-Learn
      - PyCave CPU (full data)
      - PyCave CPU (1k batches)
      - PyCave GPU (full data)
      - PyCave GPU (1k batches)
    * - 10k datapoints, 8 features, 4 clusters
      - **x1** (21.9 ms)
      - 
    * - 100k datapoints, 32 features, 16 clusters
      - **x1** (554 ms)
    * - 1M datapoints, 64 features, 64 clusters
      - **x1** (32.1 s)
