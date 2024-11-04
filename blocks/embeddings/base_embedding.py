import torch
from torch import nn

from model_configs.base import ModelConfigs


class IOEmbedding(nn.Module):
    """
    This is an Input output embedding space used inside the transformers and other architectures

    :param
    d_model -> Models dimension
    vocabulary_size -> model vocab size
    padding_idx -> padding index
    """

    def __int__(self, config: ModelConfigs):
        self.d_model = config.d_model
        self.vocabulary_size = config.vocabulary_size
        self.embeddings = nn.Embedding(num_embeddings=config.vocabulary_size,
                                       embedding_dim=config.d_model,
                                       padding_idx=config.padding_id)


    def forward(self, input_ids: torch.tensor):
        """embedding forward pass"""
        return self.embeddings(input_ids)
