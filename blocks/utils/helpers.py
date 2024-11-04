from typing import Tuple

from torch import nn


def num_parameters(model, only_trainable: bool = False, exclude_embeddings: bool = False) -> Tuple[float, float]:
    """
    Get number of (optionally, trainable or non-embeddings) parameters in the module.

    Args:
        model:
        only_trainable (`bool`, *optional*, defaults to `False`):
            Whether or not to return only the number of trainable parameters

        exclude_embeddings (`bool`, *optional*, defaults to `False`):
            Whether or not to return only the number of non-embeddings parameters

    Returns:
        `int`: The number of parameters.
    """

    if exclude_embeddings:
        embedding_param_names = [
            f"{name}.weight" for name, module_type in model.named_modules() if isinstance(module_type, nn.Embedding)
        ]
        total_parameters = [
            parameter for name, parameter in model.named_parameters() if name not in embedding_param_names
        ]
    else:
        total_parameters = list(model.parameters())

    total_trainable_numel = []
    total_numel = []
    for param in total_parameters:
        if param.requires_grad or not only_trainable:
            total_trainable_numel.append(param.numel())
        total_numel.append(param.numel())

    return sum(total_trainable_numel), sum(total_numel)
