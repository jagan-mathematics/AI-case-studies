"""model config loaders"""
from dataclasses import dataclass
from abc import ABC


@dataclass
class ModelConfigs(metaclass=ABC):
    """abstract model configuration class"""
    model_name: str
    padding_id: int
    d_model: int
    max_positions: int
    vocabulary_size: int
    use_represent_normalizer: bool
    intermediate_dim: int
    layer_norm_eps: float = 1e-05
    model_max_sequence: int = 2048
    num_heads: int = 8
    attention_dropout = 0.0
    head_dim = None


    def __post_init__(self):
        if self.head_dim is None:
            assert self.d_model % self.num_heads == 0
            self.head_dim = self.d_model // self.num_heads


    def get_padding_token(self):
        """get model padding tokens"""
        return self.padding_id
