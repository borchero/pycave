import random
import time
import numpy as np
import torch
import pycave
from pycave.bayes import GaussianMixture

# Set seed
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)

# Inputs
n = 10000
p = 2000
k = 40

# Make some Gaussian data
X = torch.randn(n, p)

# Fit PyCave GMM
gmm = GaussianMixture(
    num_components=k,
    covariance_type="full",
    init_strategy="kmeans",
    trainer_params={"enable_progress_bar": True},
    covariance_regularization=1e-3,
)
gmm = gmm.fit(X)
