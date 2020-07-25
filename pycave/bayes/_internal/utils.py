import math
import torch

def log_normal(x, means, covars):
    """
    Computes the log-probability of the given data for multiple multivariate normal distributions
    defined by their means and covariances.

    Parameters
    ----------
    x: torch.Tensor [N, D]
        The data to compute the log-probability for.
    means: torch.Tensor [K, D]
        The means of the distributions.
    covars: torch.Tensor ([K, D] or [D] or [K])
        The covariances of the distributions, depending on the covariance type. In the first case,
        each distribution is assumed to have its own diagonal covariance matrix, in the second case,
        the covariance matrix is shared among all distributions, and in the last one, the
        covariance matrix is spherical. The type of covariance is therefore inferred by the size of
        the input. If the dimension does not fit any of the documented sizes, no error will be
        thrown but the result is undefined.

    Returns
    -------
    torch.Tensor [N, K]
        The log-probabilities for every input and distribution.
    """
    num_features = x.size(1)
    precisions = 1 / covars

    if covars.size(0) == num_features: # shared diagonal covariance
        num_means = means.size(0)
        precisions = precisions.view(1, num_features).expand(
            num_means, num_features
        )

    if precisions.dim() == 2: # diagonal covariance
        cov_det = (-precisions.log()).sum(1)
        x_prob = torch.matmul(x * x, precisions.t())
        m_prob = torch.einsum('ij,ij,ij->i', means, means, precisions)
        xm_prob = torch.matmul(x, (means * precisions).t())
    else: # spherical covariance
        cov_det = -precisions.log() * num_features
        x_prob = torch.ger(torch.einsum('ij,ij->i', x, x), precisions)
        m_prob = torch.einsum('ij,ij->i', means, means) * precisions
        xm_prob = torch.matmul(x, means.t() * precisions)

    constant = math.log(2 * math.pi) * num_features
    log_prob = x_prob - 2 * xm_prob + m_prob

    return -0.5 * (constant + cov_det + log_prob)


def log_responsibilities(log_probs, comp_priors, return_log_likelihood=False):
    """
    Computes the log-responsibilities of some data based on their log-probabilities for all
    components of a gaussian mixture model and the components' weights.

    Parameters
    ----------
    log_probs: torch.Tensor [N, K]
        The log-probabilities of N datapoints for each of K distributions.
    comp_priors: torch.Tensor [K]
        The prior probabilities for each of the K distributions.
    return_log_likelihood: bool, default: False
        Whether to return the log-likelihood of observing the data given the log-probabilities and
        the component priors.

    Returns
    -------
    torch.Tensor [N, K]
        The log-responsibilities for all datapoints and distributions.
    torch.Tensor [N]
        If `return_log_likelihood` is `True`, the log-likelihood of observing each of the
        datapoints ("evidence"). To compute the log-likelihood of the entire data, sum these
        log-likelihoods up.
    """
    posterior = log_probs + comp_priors.log()
    evidence = torch.logsumexp(posterior, 1, keepdim=True)
    log_resp = posterior - evidence

    if return_log_likelihood:
        return log_resp, evidence
    return log_resp


def max_likeli_means(data, responsibilities, comp_sums=None):
    """
    Maximizes the likelihood of the data with respect to the means for a gaussian mixture model.

    Parameters
    ----------
    data: torch.Tensor [N, D]
        The data for which to maximize the likelihood.
    responsibilities: torch.Tensor [N, K]
        The responsibilities for each datapoint and component.
    comp_sums: torch.Tensor [K], default: None
        The cumulative probabilities for all components. If not given, this method can be used for
        e.g. implementing mini-batch GMM. The means are not normalized in this case.

    Returns
    -------
    torch.Tensor [K, D]
        The likelihood-maximizing means.
    """
    means = torch.matmul(responsibilities.t(), data)
    if comp_sums is not None:
        return means / comp_sums.unsqueeze(1)
    return means


def max_likeli_covars(data, responsibilities, comp_sums, means, covar_type, reg=1e-6):
    """
    Maximizes the likelihood of the data with respect to the covariances for a gaussian mixture
    model.

    Note
    ----
    Implementation is adapted from scikit-learn.

    Parameters
    ----------
    data: torch.Tensor [N, D]
        The data for which to maximize the likelihood.
    responsibilities: torch.Tensor [N, K]
        The responsibilities for each datapoint and component.
    comp_sums: torch.Tensor [K]
        The cumulative probabilities for all components.
    means: torch.Tensor [K, D]
        The means of all components.
    covar_type: str
        The type of the covariance to maximize the likelihood for. Must be one of ('diag',
        'diag-shared', 'spherical').
    reg: float, default: 1e-6
        Regularization term added to the covariance to ensure that it is positive.

    Returns
    -------
    torch.Tensor ([K, D] or [D] or [K])
        The likelihood-maximizing covariances where the shape depends on the given covariance type.
    """
    if covar_type == 'diag':
        return _max_likeli_diag_covars(data, responsibilities, comp_sums, means, reg)
    if covar_type == 'diag-shared':
        return _max_likeli_diag_covars(data, responsibilities, comp_sums, means, reg).mean(0)
    return _max_likeli_diag_covars(data, responsibilities, comp_sums, means, reg).mean(1)


def _max_likeli_diag_covars(data, responsibilities, comp_sums, means, reg):
    denom = comp_sums.unsqueeze(1)
    x_sq = torch.matmul(responsibilities.t(), data * data) / denom
    m_sq = means * means
    xm = means * torch.matmul(responsibilities.t(), data) / denom
    return x_sq - 2 * xm + m_sq + reg


def power_iteration(A, eps=1e-7, max_iterations=100):
    """
    Computes the eigenvector corresponding to the largest eigenvalue of the given square matrix.

    Parameters
    ----------
    A: torch.Tensor [N, N]
        The square matrix for which to compute the eigenvector corresponding to the largest
        eigenvalue.
    eps: float, default: 1e-7
        Convergence criterion. When the change in the vector norm is less than the given value,
        iteration is stopped.
    max_iterations: int, default: 100
        The maximum number of iterations to do when the epsilon-based convergence criterion does
        not kick in.

    Returns
    -------
    torch.Tensor [N]
        The eigenvector corresponding to the largest eigenvalue of the given square matrix.
    """
    v = torch.rand(A.size(0), device=A.device)

    for _ in range(max_iterations):
        v_old = v

        v = A.mv(v)
        v = v / v.norm()

        if (v - v_old).norm() < eps:
            break

    return v


def to_one_hot(X, n):
    """
    Creates a one-hot matrix from a set of indices.

    Parameters
    ----------
    X: torch.Tensor [N, D]
        The indices to convert into one-hot vectors.
    n: int
        The number of entries in the one-hot vectors.

    Returns
    -------
    torch.Tensor [N, D, n]
        The one-hot matrix.
    """
    return torch.eye(n, device=X.device)[X]


def normalize(value, dims=-1, return_denom=False):
    """
    Normalizes a tensor in the given dimension such that all entries sum to one. It can then be
    interpreted as a probability distribution.

    Parameters
    ----------
    value: torch.Tensor [...]
        A tensor of arbitrary shape.
    dims: int or list of int, default: -1
        The dimension which to sum, i.e. which dimensions model a probability distribution.
    return_denom: bool, default: False
        Whether to return the denominator that is used to normalize values.

    Returns
    -------
    torch.Tensor [...]
        The normalized input tensor (same shape as the input).
    torch.Tensor [...]
        The normalizing denominator if `return_denom` is `True` (same shape as the input).
    """
    denom = value.sum(dims, keepdim=True)
    if return_denom:
        return value / denom, denom
    return value / denom


def batch_weights(num_datapoints, batch_size):
    """
    Computes the weights for each of the batches of the given batch size needed to fully represent
    the data.

    Parameters
    ----------
    num_datapoints: int
        The number of datapoints to fit in the batches.
    batch_size: int
        The number of datapoints per batch.

    Returns
    -------
    int
        The number of batches.
    torch.Tensor [N]
        A tensor with weights for each batch where the number of entries N times the batch size is
        greater than or equal to the number of datapoints.
    """
    num_batches = int(math.ceil(num_datapoints / batch_size))
    weights = torch.ones(num_batches)

    # 1.1) Adjust weight of last batch
    last_batch_size = num_datapoints - batch_size * (weights.size(0) - 1)
    weights[-1] = last_batch_size / batch_size

    return num_batches, weights / weights.sum()


def packed_get_first(sequences, batch_sizes):
    """
    Retrieves the first item of each sequence in the given data according to the given batch sizes.

    Parameters
    ----------
    sequences: torch.Tensor [N, ...]
        The data tensor of the packed sequences (number of total elements N).
    batch_sizes: torch.Tensor [B]
        The batch sizes (maximum sequence length B).

    Returns
    -------
    torch.Tensor [S, ...]
        The data of the packed sequences with each sequence's first element (number of sequences S).
    """
    num_sequences = batch_sizes[0].item()
    return sequences[:num_sequences]


def packed_get_last(sequences, batch_sizes):
    """
    Retrieves the last item of each sequence in the given data according to the given batch sizes.

    Parameters
    ----------
    sequences: torch.Tensor [N, ...]
        The data tensor of the packed sequences (number of total elements N).
    batch_sizes: torch.Tensor [B]
        The batch sizes (maximum sequence length B).

    Returns
    -------
    torch.Tensor [S, ...]
        The data of the packed sequences with each sequence's last element (number of sequences S).
    """
    num_sequences = batch_sizes[0].item()
    max_length = batch_sizes.size(0)
    device = sequences.device

    result = torch.empty(num_sequences, *sequences.size()[1:], device=device)
    result_offset = 0
    data_offset = sequences.size(0)

    for i in range(max_length - 1, -1, -1):
        prev_count = batch_sizes[i+1].item() if i < max_length - 1 else 0
        count = batch_sizes[i].item()
        diff = count - prev_count

        result[result_offset: result_offset+diff] = sequences[data_offset-diff: data_offset]

        result_offset += diff
        data_offset -= count

    return result

def packed_drop_first(sequences, batch_sizes):
    """
    Drops the first item of each sequence in the given data according to the given batch sizes.

    Parameters
    ----------
    sequences: torch.Tensor [N, ...]
        The data tensor of the packed sequences (number of total elements N).
    batch_sizes: torch.Tensor [B]
        The batch sizes (maximum sequence length B).

    Returns
    -------
    torch.Tensor [N - S, ...]
        The data of the packed sequences where each sequence's first element has been dropped
        (number of sequences S).
    """
    num_sequences = batch_sizes[0].item()
    return sequences[num_sequences:]

def packed_drop_last(sequences, batch_sizes):
    """
    Drops the last item of each sequence in the given data according to the given batch sizes.

    Parameters
    ----------
    sequences: torch.Tensor [N, ...]
        The data tensor of the packed sequences (number of total elements N).
    batch_sizes: torch.Tensor [B]
        The batch sizes (maximum sequence length B).

    Returns
    -------
    torch.Tensor [N - S, ...]
        The data of the packed sequences where each sequence's last element has been dropped (number
        of sequences S).
    """
    num_sequences = batch_sizes[0].item()
    max_length = batch_sizes.size(0)
    device = sequences.device

    result = torch.empty(sequences.size(0) - num_sequences, *sequences.size()[1:], device=device)
    result_offset = result.size(0)
    data_offset = sequences.size(0) - batch_sizes[-1].item()

    for i in range(max_length - 2, -1, -1):
        prev_count = batch_sizes[i+1].item()
        count = batch_sizes[i].item()

        result[result_offset-prev_count: result_offset] = \
            sequences[data_offset-count: data_offset-count+prev_count]

        result_offset -= prev_count
        data_offset -= count

    return result
