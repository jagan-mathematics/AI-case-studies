import torch
from torch import nn

from blocks.decoder.PNRDecoder import PNRDecoderBlock
from blocks.layers.norms import RMSNorm
from model_configs.base import ModelConfigs


def _prepare_4d_causal_attention_mask_with_cache_position(
    attention_mask: torch.Tensor,
    sequence_length: int,
    target_length: int,
    dtype: torch.dtype,
    device: torch.device,
    min_dtype: float,
    batch_size: int,
):
    """
    Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

    Args:
        attention_mask (`torch.Tensor`):
            A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
        sequence_length (`int`):
            The sequence length being processed.
        target_length (`int`):
            The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
        dtype (`torch.dtype`):
            The dtype to use for the 4D attention mask.
        device (`torch.device`):
            The device to plcae the 4D attention mask on.
        min_dtype (`float`):
            The minimum value representable with the dtype `dtype`.
        batch_size (`torch.Tensor`):
            Batch size.
    """
    if attention_mask is not None and attention_mask.dim() == 4:
        # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
        causal_mask = attention_mask
    else:
        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device)

        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )
    return causal_mask


def initialize_weights(module, std=0.02):
    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_uniform(module.weight)
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=std)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, RMSNorm):
        module.weight.data.zero_()

def _update_causal_mask(
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        output_attentions: bool,
):
    dtype, device = input_tensor.dtype, input_tensor.device
    min_dtype = torch.finfo(dtype).min
    sequence_length = input_tensor.shape[1]

    target_length = (
        attention_mask.shape[-1]
        if isinstance(attention_mask, torch.Tensor)
        else sequence_length + 1
    )

    # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
    causal_mask = _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask,
        sequence_length=sequence_length,
        target_length=target_length,
        dtype=dtype,
        device=device,
        min_dtype=min_dtype,
        batch_size=input_tensor.shape[0],
    )

    return causal_mask


class Model(nn.Module):
    def __init__(self, config: ModelConfigs):
        super().__init__()
        self.padding_idx = config.padding_id
        self.vocab_size = config.vocabulary_size
        self.hidden_size = config.d_model

        self.num_hidden_layers = config.num_layers

        self.embed_tokens = nn.Embedding(self.vocab_size, self.hidden_size, self.padding_idx)

        self.layers = nn.ModuleList(
            [PNRDecoderBlock(config, layer_idx) for layer_idx in range(self.num_hidden_layers)]
        )

        self.norm = RMSNorm(self.hidden_size, eps=1e-6)

        self.apply(initialize_weights)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def get_layers(self, idx=None):
        return self.layers if idx is None else self.layers[idx]

    def forward(self, input_ids,
                attention_mask,
                output_attentions=False,
                output_hidden_states=False):
        inputs_embeds = self.embed_tokens(input_ids) # B x S x D
        position_ids = torch.arange(
            inputs_embeds.shape[1], device=inputs_embeds.device
            )

        position_ids = position_ids.unsqueeze(0)

        hidden_states = inputs_embeds
        normalizer = torch.tensor(self.config.hidden_size ** 0.5, dtype=hidden_states.dtype)
        hidden_states *= normalizer

        causal_mask = _update_causal_mask(
            attention_mask, inputs_embeds, output_attentions
        )

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                output_attentions=output_attentions
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions += (layer_outputs[1],)
        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return {
            "last_hidden_state": hidden_states,
            "all_hidden_state": all_hidden_states,
            "all_self_attentions": all_self_attentions
        }