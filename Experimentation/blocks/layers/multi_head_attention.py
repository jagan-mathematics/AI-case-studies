import math

import torch
from torch import nn

from Experimentation.blocks.layers.base_attention import BaseAttention


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

