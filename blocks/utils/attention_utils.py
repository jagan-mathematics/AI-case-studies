from typing import Tuple

import torch
from torch import Tensor



def get_extended_attention_mask(
    attention_mask: Tensor, input_shape: Tuple[int], device: torch.device = None, dtype: torch.float = None
) -> Tensor:
    """
    Makes broadcast attention for encoder layer

    Arguments:
        attention_mask (`torch.Tensor`):
            Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
        input_shape (`Tuple[int]`):
            The shape of the input to the model.
        device:
            token device
        dtype:
            data type of the source token

    Returns:
        `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
    """

    if device is None:
        device = attention_mask.device

    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is a decoder, apply a causal mask in addition to the padding mask
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]

        extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(
            f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
        )

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and the type's smallest value for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
    if extended_attention_mask.device != device:
        extended_attention_mask = extended_attention_mask.to(device)

    return extended_attention_mask


def create_extended_casual_attention_mask_for_decoder(input_shape, attention_mask, device=None):
    """method to create extended attention mask for decoder layer"""

    if device is None:
        device = attention_mask.device
    batch_size, seq_length = input_shape

    seq_ids = torch.arange(seq_length, device=device)
    causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
    # in case past_key_values are used we need to add a prefix ones mask to the causal mask
    # causal and attention masks must have same type with pytorch version < 1.3
    causal_mask = causal_mask.to(attention_mask.dtype)

    extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
    extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(attention_mask.dtype).min
    if extended_attention_mask.device != device:
        extended_attention_mask = extended_attention_mask.to(device)
    return extended_attention_mask


def create_extended_cross_attention_mask_for_decoder(batch_size, sequence_length, target_length, encoder_attention_mask, dtype, device):
    if device is None:
        device = encoder_attention_mask.device

    if dtype is None:
        dtype = encoder_attention_mask.dtype

    casual_mask = torch.full((sequence_length, target_length), fill_value=1, device=device, dtype=dtype)
    casual_mask = casual_mask[None, None, :, :].expand(batch_size, 1, -1, -1)

    if len(encoder_attention_mask.size) == 2:
        padding_mask = encoder_attention_mask[:, None, None, :]
    elif len(encoder_attention_mask.size) == 4:
        padding_mask = encoder_attention_mask
    else:
        raise ValueError(
            f"wrong attention_mask (shape {encoder_attention_mask.shape})"
        )

    casual_mask = (casual_mask * padding_mask).to(dtype)
    casual_mask = (1.0 - casual_mask) * torch.finfo(dtype).min
    return casual_mask

