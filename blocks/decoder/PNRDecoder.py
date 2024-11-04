from torch import nn

from blocks.layers.point_wise_projection import PointWiseGateProjection
from model_configs.base import ModelConfigs
from blocks.layers.norms import RMSNorm
from blocks.layers.multi_head_attention import RopeAttention


class PNRDecoderBlock(nn.Module):
    """
    Pre residual norm connection as in GPT and Gemma
    using RMS norm for stable training with scale invariant property
    """
    def __init__(self, config: ModelConfigs, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        self.hidden_size = config.d_model

        self.mlp = PointWiseGateProjection(config)
        self.attention_layer = RopeAttention(config=config)
        self.input_norm = RMSNorm(dim=self.hidden_size, eps=1e-06)
        self.post_attention_norm = RMSNorm(dim=self.hidden_size, eps=1e-06)

    def forward(self, hidden_state,
                attention_mask,
                position_ids,
                output_attentions=False):

        """
        https://arxiv.org/pdf/2002.04745 (PRE-Norm)

        x = embedding of each tokens (B x S x D)
        a_x = self_attention autput (B x S x D)

        Self-attention sub-block 1
        x => norm(x) => n_x => att(n_x) => (a_x, a_score) => a_x + x => h_x

        MLP sub-block: 2

        h_x => norm(h_x) => n_hx => mlp(n_hx) => n_x => x_x + h_x => h_x
        """
        # casual attention block
        residual_x = hidden_state

        hidden_state = self.input_norm(hidden_state)

        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_state,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
        )

        hidden_states = residual_x + hidden_state

        # point forward inner bloc
        residual_x = hidden_state

        hidden_state = self.post_attention_norm(hidden_states)
        hidden_state = self.mlp(hidden_state)
        hidden_state = residual_x + hidden_state

        if output_attentions:
            return hidden_state, self_attn_weights
        return hidden_state

