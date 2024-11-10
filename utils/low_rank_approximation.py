import torch
import torch.nn as nn


def decompose_linear_layer(linear_layer, rank):
    # Extract weight and bias from the original MLP (fully connected) layer
    W = linear_layer.weight.data
    bias = linear_layer.bias.data if linear_layer.bias is not None else None

    # Perform SVD on the weight matrix
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)

    # Truncate to the specified rank for low-rank approximation
    U_k = U[:, :rank]
    S_k = torch.diag(S[:rank])
    Vt_k = Vt[:rank, :]

    # Create new weights for the two consecutive layers
    first_weight = U_k @ S_k
    second_weight = Vt_k

    # First layer: reduces dimensionality
    first_layer = nn.Linear(W.size(1), rank, bias=True)
    first_layer.weight.data = first_weight

    # Second layer: restores dimensionality
    second_layer = nn.Linear(rank, W.size(0), bias=False)
    second_layer.weight.data = second_weight
    if bias is not None:
        first_layer.bias.data = bias

    return nn.Sequential(second_layer, first_layer)


def apply_lra_to_model(model, rank):
    """
    Takes in a pre-trained model and replaces MLP layers in each transformer block with low-rank approximations.

    Args:
    - model (torch.nn.Module): Pre-trained Vision Transformer model.
    - rank (int): The desired rank for the low-rank approximation.

    Returns:
    - torch.nn.Module: The modified ViT model with low-rank approximations in place of original MLP layers.
    """

    def convert_block(module):
        """
        Recursively applies low-rank approximation to all MLP layers in a module.
        """
        for name, child in module.named_children():
            # If the layer is a block (e.g., encoder layer in ViT), apply LRA to MLP within it
            if isinstance(child, nn.Linear) and "fc" in name:
                # If it's a standalone linear layer, replace with LRA layers
                setattr(module, name, decompose_linear_layer(child, rank))

            else:
                # Recursively call convert_block for other container layers
                convert_block(child)

    # Apply low-rank approximation transformation to each block in the model
    convert_block(model)

    return model

