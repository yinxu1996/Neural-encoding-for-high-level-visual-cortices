import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

def collate_fn(data, max_len=None):
    batch_size = len(data)
    features, labels = zip(*data)

    # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
    lengths = [
        X.shape[0] for X in features
    ]  # original sequence length for each time series
    if max_len is None:
        max_len = max(lengths)

    X = torch.zeros(
        batch_size, max_len, features[0].shape[-1]
    )  # (batch_size, padded_length, feat_dim)
    for i in range(batch_size):
        end = min(lengths[i], max_len)
        X[i, :end, :] = features[i][:end, :]

    targets = torch.stack(labels, dim=0)  # (batch_size, num_labels)

    padding_masks = padding_mask(
        torch.tensor(lengths, dtype=torch.int16), max_len=max_len
    )  # (batch_size, padded_length) boolean tensor, "1" means keep

    return X, targets, padding_masks

def padding_mask(lengths, max_len=None):
    """
    Used to mask padded positions: creates a (batch_size, max_len) boolean mask from a tensor of sequence lengths,
    where 1 means keep element at this position (time step)
    """
    batch_size = lengths.numel()
    max_len = (
        max_len or lengths.max_val()
    )  # trick works because of overloading of 'or' operator for non-boolean types
    return (
        torch.arange(0, max_len, device=lengths.device)
        .type_as(lengths)  # convert to same type as lengths tensor
        .repeat(batch_size, 1)  # (batch_size, max_len)
        .lt(lengths.unsqueeze(1))
    )

def normalize_ts(ts):
    """normalize a time-series data

    Args:
        ts (numpy.ndarray): The input time-series in shape (timestamps, feature).

    Returns:
        ts (numpy.ndarray): The processed time-series.
    """
    scaler = StandardScaler()
    scaler.fit(ts)
    ts = scaler.transform(ts)
    return ts


def normalize_batch_ts(batch):
    """normalize a batch of time-series data

    Args:
        batch (numpy.ndarray): A batch of input time-series in shape (n_samples, timestamps, feature).

    Returns:
        A batch of processed time-series.
    """
    return np.array(
        list(map(normalize_ts, batch))
    )