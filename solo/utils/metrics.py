# Copyright 2023 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from typing import Dict, List, Sequence, Tuple

import torch


def accuracy_at_k(
    outputs: torch.Tensor, targets: torch.Tensor, top_k: Sequence[int] = (1, 5)
) -> Sequence[int]:
    """Computes the accuracy over the k top predictions for the specified values of k.

    Args:
        outputs (torch.Tensor): output of a classifier (logits or probabilities).
        targets (torch.Tensor): ground truth labels.
        top_k (Sequence[int], optional): sequence of top k values to compute the accuracy over.
            Defaults to (1, 5).

    Returns:
        Sequence[int]:  accuracies at the desired k.
    """

    with torch.no_grad():
        maxk = max(top_k)
        batch_size = targets.size(0)

        _, pred = outputs.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        res = []
        for k in top_k:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def weighted_mean(outputs: List[Dict], key: str, batch_size_key: str) -> float:
    """Computes the mean of the values of a key weighted by the batch size.

    Args:
        outputs (List[Dict]): list of dicts containing the outputs of a validation step.
        key (str): key of the metric of interest.
        batch_size_key (str): key of batch size values.

    Returns:
        float: weighted mean of the values of a key
    """

    value = 0
    n = 0
    for out in outputs:
        value += out[batch_size_key] * out[key]
        n += out[batch_size_key]
    value = value / n
    return value.squeeze(0)

#############
# Sparsity Metrics
#############

def l1_sparsity_metric(val_feats: torch.Tensor, eps: float = 1e-12) -> float:
    """
    Calculates the mean l1 sparsity metric: (1/D) * (||z_row||_1 / ||z_row||_2)^2
    """
    with torch.no_grad():
        D = val_feats.shape[1]
        l1_norms = torch.linalg.norm(val_feats, ord=1, dim=1)
        l2_norms = torch.linalg.norm(val_feats, ord=2, dim=1)
        l1_sparsity_per_sample = (1.0 / D) * (l1_norms / (l2_norms + eps))**2
        return l1_sparsity_per_sample.mean().item()

def l0_sparsity_metric(val_feats: torch.Tensor) -> float:
    """
    Calculates the mean l0 sparsity metric: fraction of nonzero elements per sample
    """
    with torch.no_grad():
        D = val_feats.shape[1]
        l0_sparsity_per_sample = (val_feats != 0).sum(dim=1).float() / D
        return l0_sparsity_per_sample.mean().item()

def variance_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Computes variance loss given batch of projected features z1 from view 1 and
    projected features z2 from view 2.
    """
    eps = 1e-4
    std_z1 = torch.sqrt(z1.var(dim=0) + eps)
    std_z2 = torch.sqrt(z2.var(dim=0) + eps)
    std_loss = torch.mean(torch.relu(1 - std_z1)) + torch.mean(torch.relu(1 - std_z2))
    return std_loss

def covariance_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Computes covariance loss given batch of projected features z1 from view 1 and
    projected features z2 from view 2.
    """
    N, D = z1.size()
    z1 = z1 - z1.mean(dim=0)
    z2 = z2 - z2.mean(dim=0)
    cov_z1 = (z1.T @ z1) / (N - 1)
    cov_z2 = (z2.T @ z2) / (N - 1)
    diag = torch.eye(D, device=z1.device)
    cov_loss = cov_z1[~diag.bool()].pow_(2).sum() / D + cov_z2[~diag.bool()].pow_(2).sum() / D
    return cov_loss

# Deprecated/Legacy aliases
def embedding_sparsity_metric(embeddings: torch.Tensor, epsilon: float = 1e-12) -> Tuple[float, float, float]:
    """Legacy alias for l1_sparsity_metric (returning Tuple for compatibility)"""
    val = l1_sparsity_metric(embeddings, epsilon)
    return val, val, val

def count_avg_nonzero_elements_per_sample(tensor: torch.Tensor) -> float:
    """Legacy alias for l0_sparsity_metric"""
    return l0_sparsity_metric(tensor)

def batch_sparsity_metric(tensor_data: torch.Tensor, epsilon: float = 1e-12) -> Tuple[float, float, float]:
    if not isinstance(tensor_data, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor.")
    if tensor_data.ndim != 2:
        raise ValueError("Input tensor must be 2-dimensional (B x D).")
    B, D = tensor_data.shape
    l1_norm_per_column = torch.linalg.norm(tensor_data, ord=1, dim=0)
    l2_norm_sq_per_column = torch.linalg.norm(tensor_data, ord=2, dim=0)**2
    sparsity_metric_per_column = (l1_norm_per_column**2) / (l2_norm_sq_per_column + epsilon)
    sparsity_metric_per_column = sparsity_metric_per_column / B
    return sparsity_metric_per_column.max().item(), sparsity_metric_per_column.mean().item(), sparsity_metric_per_column.min().item()

def count_avg_nonzero_elements_per_dimension(tensor: torch.Tensor) -> float:
    return ((tensor != 0).sum(dim=0) / tensor.shape[0]).mean().item()

def active_feature_fraction(tensor_data: torch.Tensor, threshold: float = 1e-3) -> float:
    active_mask = (tensor_data.abs() > threshold)
    return active_mask.sum().item() / tensor_data.numel()
