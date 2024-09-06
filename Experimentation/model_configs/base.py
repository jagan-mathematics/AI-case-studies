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
    layer_norm_eps: float = 1e-05

    def get_padding_token(self):
        """get model padding tokens"""
        return self.padding_id
