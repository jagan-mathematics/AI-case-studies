import math

import torch
from torch import nn

from Experimentation.blocks.layers.base_attention import BaseAttention
from Experimentation.blocks.positional_encoding.rope import RotaryEmbedding, apply_rotary_pos_emb
from Experimentation.model_configs.base import ModelConfigs


class MultiHeadAttention(BaseAttention):
    """
    MultiHeadAttention Module

    B => Batch_Size
    L => Sequence length
    D => Model Dim
    """

    def forward(self, x, attention_mask=None):
        """
        :param x: input hidden state (B x L x D)
        :type x: torch.tensor
        :param attention_mask: mask mask
        :type attention_mask: torch.tensor
        """
        batch_size, q_len, _ = x.size()

        query: torch.tensor = self.query_projection(x)  # B x L x D => B x L x (H * H_d)
        key: torch.tensor = self.key_projection(x)  # B x L x D => B x L x (H * H_d)
        value: torch.tensor = self.value_projection(x)  # B x L x D => B x L x (H * H_d)

        # B x L x D => B x H x L x H_d
        query = query.view(batch_size, q_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        key = key.view(batch_size, q_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        value = value.view(batch_size, q_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # (B x H x L x H_d) @ (B x H x H_d x L) => (B x H x L x L)
        attention_scores = torch.matmul(query, key.view(-1, -2))
        attention_scores /= math.sqrt(self.num_heads * self.head_dim)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_scores = nn.functional.softmax(attention_scores, dim=-1)


        attention_scores = nn.functional.dropout(attention_scores, p=self.attention_dropout, training=self.training)


        content = torch.matmul(attention_scores, value)  # (B x H x L x L) @ (B x H x L x H_d) => (B x H x L x H_d)
        if content.size() != (batch_size, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, q_len, self.head_dim)}, but is"
                f" {content.size()}"
            )

        content = content.transpose(1, 2).contiguous()  # (B x L x H x H_d)

        content = content.view(batch_size, q_len, -1)
        content = self.o_proj(content)

        return content, attention_scores



class RopeAttention(nn.Module):
    """
    Attention with rope embedding
    """
    def __init__(self, config: ModelConfigs):
        super().__init__()
        assert self.head_dim is not None
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.attention_dropout = config.attention_dropout
        self.head_dim = config.head_dim

        self.num_heads = config.num_heads
        self.attention_dim = self.num_heads * self.head_dim

        self.query_projection = nn.Linear(self.d_model, self.attention_dim, bias=False)
        self.key_projection = nn.Linear(self.d_model, self.attention_dim, bias=False)
        self.value_projection = nn.Linear(self.d_model, self.attention_dim, bias=False)

        self.output_projection = nn.Linear(self.attention_dim, self.d_model, bias=False)

        self.rope = RotaryEmbedding(dim=self.head_dim,
                                    max_position_embeddings=config.model_max_sequence,
                                    base=10_000.0)

        self.scaling = 1 / math.sqrt(config.head_dim)

    def forward(self, hidden_states,
                attention_mask,
                position_ids,
                output_attentions=False):
        b_size, seq_len, _ = hidden_states.size()

        query_state = self.query_projection(hidden_states)
        key_state = self.key_projection(hidden_states)
        value_state = self.value_projection(hidden_states)

        query_state = query_state.view(b_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_state = key_state.view(b_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_state = value_state.view(b_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_state, position_ids)
        query_state, key_state = apply_rotary_pos_emb(query_state, key_state, cos, sin)

        attn_weights = torch.matmul(query_state, key_state.transpose(2, 3)) * self.scaling

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_state.shape[-2]]  # B x H x Q_s x K_s
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_state.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_state)

        if attn_output.size() != (b_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(b_size, self.num_heads, seq_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.view(b_size, seq_len, -1)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights
