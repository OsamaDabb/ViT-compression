import torch
import torch.nn as nn
import numpy as np


def unstructured_magnitude_prune(model, prune_percentage):

    if prune_percentage == 0:
        return

    mask = {}

    for name, param in model.named_parameters():
        if "weight" in name:  # apply to linear weights, etc.
            # Flatten the parameter and sort its absolute values
            flat_param = param.view(-1)
            sorted_params, _ = torch.sort(flat_param.abs())

            # Find the threshold value that corresponds to the prune percentage
            prune_idx = int(prune_percentage * len(sorted_params))
            threshold = sorted_params[prune_idx]

            # Apply threshold-based pruning
            param.data = torch.where(param.abs() < threshold, torch.zeros_like(param), param.data)
            mask[name] = param != 0

    return mask


def column_magnitude_pruning(model, pruning_ratio=0.2):
    """
    Applies column-based structured magnitude pruning on each layer in the model.

    Args:
        model (nn.Module): The model to prune.
        pruning_ratio (float): Fraction of columns to prune per layer.
    """

    mask = {}

    for name, param in model.named_parameters():
        if "weight" in name:  # Modify as needed to include other layer types
            # Compute the L2 norm of each column
            column_norms = torch.norm(param, p=2, dim=0)  # Shape: [in_features]

            # Determine the number of columns to prune
            num_columns_to_prune = int(pruning_ratio * weight.size(1))

            # Find the indices of the columns with the smallest norms
            _, prune_indices = torch.topk(column_norms, num_columns_to_prune, largest=False)

            # Set the selected columns to zero
            weight[:, prune_indices] = 0  # Prune columns

            mask[name] = weight != 0

    return mask


# zeroes the value and gradient of masked values
def apply_mask(model, mask):

    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in mask:
                param.data *= mask[name]
                param.grad *= mask[name]
