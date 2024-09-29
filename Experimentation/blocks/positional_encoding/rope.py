# https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma/modeling_gemma.py#L76
import torch
from torch import nn


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """
    Applies Rotary Position Embedding to the query and key tensors.

    The input vecor will be split into two chunks q => x1, x2 where x1, x2 belongs to real space of dimension d//2

    rotate_half => (-x2, x1)

    then the final calculation will be processed as [(x1 * cos + (-x2) * sin), (x2 * cos + x1 * sin)]
    """
    # unsqueeze at header axis
    if len(cos.size()) == 3 and len(q.size() == 4):
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, scaling_factor=None, dynamic_scale=False):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        assert ((dynamic_scale and scaling_factor is not None) and scaling_factor <= 1) or not dynamic_scale
        self.scaling_factor = scaling_factor
        self.dynamic_scale = dynamic_scale
        angular_freq = 1.0 / (
                    self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))  # (d/2, )
        self.register_buffer("angular_freq", tensor=angular_freq, persistent=False)

    @torch.no_grad()
    def re_initialize_angular_freq(self, seq_len):
        if seq_len > self.max_position_embeddings:
            base = self.base * (
                    (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            angular_freq = 1.0 / (
                    base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(x.device) / self.dim)
            )
            self.register_buffer("angular_freq", angular_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        """
        x -> (b x h x s x d)
        positions_ids = (1 x s)
        """
        seq_len = x.shape[0]
        if self.dynamic_scale:
            self.re_initialize_angular_freq(seq_len)

        if self.angular_freq.device != x.device:
            self.angular_freq.to(x.device)

        # (b x d/2 x 1)
        inv_freq_expanded = self.angular_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)

        # (b x 1 x s)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            # (b x d/2 x 1) @ (b x 1 x s) -> (b x s x d/2)
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            # (b x s x d)
            emb = torch.cat((freqs, freqs), dim=-1)

            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
